# cb_app/forms.py
from django import forms
from django.contrib.auth.models import User



class SignUpForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ("username", "email", "password")

    def clean(self):
        cleaned = super().clean()
        if cleaned.get("password") != cleaned.get("confirm_password"):
            self.add_error("confirm_password", "Passwords do not match")
        return cleaned


class UploadFileForm(forms.Form):
    excel_file = forms.FileField()


