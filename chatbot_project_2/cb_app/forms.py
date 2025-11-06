# cb_app/forms.py
from django import forms
from django.contrib.auth.models import User
from .models import RegisteredTicket


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


# ✅ New: Form for the “Register Ticket” feature
class RegisteredTicketForm(forms.ModelForm):
    class Meta:
        model = RegisteredTicket
        fields = ["subject", "category", "description", "priority"]

        widgets = {
            "subject": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "Enter ticket subject"
            }),
            "category": forms.TextInput(attrs={
                "class": "form-control",
                "placeholder": "Enter issue category"
            }),
            "description": forms.Textarea(attrs={
                "class": "form-control",
                "placeholder": "Describe your issue here...",
                "rows": 4
            }),
            "priority": forms.Select(attrs={"class": "form-select"}),
        }

        labels = {
            "subject": "Ticket Subject",
            "category": "Category",
            "description": "Description",
            "priority": "Priority",
        }
