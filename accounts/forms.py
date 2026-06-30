from django import forms
from django.conf import settings
from django.contrib.auth.forms import PasswordChangeForm

from .models import Organization, User


class AdminUserCreationForm(forms.ModelForm):
    """관리자 전용 사용자 생성 폼.

    관리자는 비밀번호를 직접 입력하지 않고, 시스템이 초기 비밀번호를 설정한다.
    최초 로그인 사용자는 반드시 비밀번호를 변경해야 한다.

    이 서비스는 세분화된 Django 권한 그룹(groups)을 사용하지 않고,
    "관리자(is_superuser) / 일반사용자" 2단계로만 구분한다.
    is_staff(장고 admin 사이트 접근 권한)는 사용하지 않으므로 폼에 노출하지 않는다.
    """

    is_superuser = forms.BooleanField(
        label="관리자 권한 부여",
        required=False,
        help_text="체크하면 전체 부서의 모든 문서를 조회/관리할 수 있는 관리자 계정이 됩니다.",
    )

    class Meta:
        model = User
        fields = (
            "username",
            "last_name",
            "first_name",
            "email",
            "phone",
            "position",
            "organization",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["organization"].queryset = Organization.objects.filter(is_active=True)
        self.fields["organization"].required = False
        self.fields["organization"].empty_label = "선택하세요"

    def clean_username(self):
        username = self.cleaned_data["username"].strip()
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("이미 사용 중인 아이디입니다.")
        return username

    def clean(self):
        cleaned_data = super().clean()
        is_superuser = cleaned_data.get("is_superuser")
        organization = cleaned_data.get("organization")
        if not is_superuser and not organization:
            self.add_error("organization", "일반사용자 계정은 소속부서를 반드시 선택해야 합니다.")
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.is_superuser = self.cleaned_data.get("is_superuser", False)
        # is_staff(장고 admin 사이트 로그인)는 이 서비스에서 사용하지 않으므로 항상 False로 둔다.
        user.is_staff = False
        user.set_initial_password()
        if commit:
            user.save()
        return user


class AccountEditForm(forms.ModelForm):
    """관리자 전용 사용자 정보 수정 폼.

    아이디(username)는 변경하지 않으며, 비밀번호도 이 폼에서는 다루지 않는다
    (비밀번호 재설정은 잠금/해제 또는 별도 절차로 처리).
    """

    is_superuser = forms.BooleanField(
        label="관리자 권한 부여",
        required=False,
        help_text="체크하면 전체 부서의 모든 문서를 조회/관리할 수 있는 관리자 계정이 됩니다.",
    )

    class Meta:
        model = User
        fields = (
            "last_name",
            "first_name",
            "email",
            "phone",
            "position",
            "organization",
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["organization"].queryset = Organization.objects.filter(is_active=True)
        self.fields["organization"].required = False
        self.fields["organization"].empty_label = "선택하세요"
        if self.instance and self.instance.pk:
            self.fields["is_superuser"].initial = self.instance.is_superuser

    def clean(self):
        cleaned_data = super().clean()
        is_superuser = cleaned_data.get("is_superuser")
        organization = cleaned_data.get("organization")
        if not is_superuser and not organization:
            self.add_error("organization", "일반사용자 계정은 소속부서를 반드시 선택해야 합니다.")
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        user.is_superuser = self.cleaned_data.get("is_superuser", False)
        if commit:
            user.save()
        return user


class OrganizationCreateForm(forms.ModelForm):
    class Meta:
        model = Organization
        fields = ["name", "code"]
        labels = {"name": "부서명", "code": "부서코드"}


class OrganizationEditForm(forms.ModelForm):
    class Meta:
        model = Organization
        fields = ["name", "code", "is_active"]
        labels = {"name": "부서명", "code": "부서코드", "is_active": "사용 여부"}


class WorkitPasswordChangeForm(PasswordChangeForm):
    """사용자 비밀번호 변경 폼.

    Django settings.AUTH_PASSWORD_VALIDATORS를 그대로 사용한다.
    """

    def clean_new_password1(self):
        password = self.cleaned_data.get("new_password1")
        initial_password = getattr(settings, "ACCOUNTS_INITIAL_PASSWORD", "Workit2026!")
        if password == initial_password:
            raise forms.ValidationError("초기 비밀번호와 동일한 비밀번호는 사용할 수 없습니다.")
        return password