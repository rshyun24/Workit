import re

from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _


class EnglishNumberSpecialCharacterValidator:
    """비밀번호에 영문, 숫자, 특수문자가 모두 포함되도록 검증한다."""

    def validate(self, password, user=None):
        has_english = re.search(r"[A-Za-z]", password or "")
        has_number = re.search(r"\d", password or "")
        has_special = re.search(r"[^A-Za-z0-9]", password or "")

        if not (has_english and has_number and has_special):
            raise ValidationError(
                _("비밀번호는 영문, 숫자, 특수문자를 모두 포함해야 합니다."),
                code="password_no_english_number_special",
            )

    def get_help_text(self):
        return _("비밀번호는 영문, 숫자, 특수문자를 모두 포함해야 합니다.")
