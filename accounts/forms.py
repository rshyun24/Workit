from django import forms
from django.conf import settings
from django.contrib.auth.forms import PasswordChangeForm

from .models import User


class AdminUserCreationForm(forms.ModelForm):
    """관리자 전용 사용자 생성 폼.

    관리자는 비밀번호를 직접 입력하지 않고, 시스템이 초기 비밀번호를 설정한다.
    최초 로그인 사용자는 반드시 비밀번호를 변경해야 한다.
    """

    class Meta:
        model = User
        fields = (
            "username",
            "last_name",
            "first_name",
            "email",
            "phone",
            "department",
            "position",
            "organization",
            "is_active",
            "is_staff",
            "is_superuser",
            "groups",
        )
        widgets = {
            "groups": forms.CheckboxSelectMultiple,
        }

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_initial_password()
        if commit:
            user.save()
            self.save_m2m()
        return user


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
