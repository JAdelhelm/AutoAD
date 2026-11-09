from __future__ import annotations

from django import forms
from django.conf import settings

from .services.readers import ENCODINGS, SUPPORTED_SUFFIXES, ReadError, read_table
from .services.runner import DEFAULT_SETTINGS, DETECTORS, ROLES

SEPARATOR_CHOICES = [
    ("auto", "Detect"),
    ("comma", "Comma ,"),
    ("semicolon", "Semicolon ;"),
    ("tab", "Tab"),
    ("pipe", "Pipe |"),
]
DECIMAL_CHOICES = [("auto", "Detect"), ("point", "Point 1.5"), ("comma", "Comma 1,5")]
ENCODING_CHOICES = [(value, "Detect" if value == "auto" else value) for value in ENCODINGS]


class TableUploadForm(forms.Form):
    """A file plus the options to read it. After validation ``frame`` holds the table."""

    file = forms.FileField(
        widget=forms.ClearableFileInput(attrs={"accept": ",".join(sorted(SUPPORTED_SUFFIXES))}),
        help_text="CSV, TSV, TXT, Excel (.xlsx), JSON, JSON Lines or Parquet.",
    )
    name = forms.CharField(max_length=200, required=False, help_text="Defaults to the file name.")
    separator = forms.ChoiceField(choices=SEPARATOR_CHOICES, initial="auto", help_text="Text files only.")
    decimal = forms.ChoiceField(
        choices=DECIMAL_CHOICES, initial="auto", label="Decimal mark", help_text="Text files only."
    )
    encoding = forms.ChoiceField(choices=ENCODING_CHOICES, initial="auto", help_text="Text files only.")
    sheet = forms.CharField(
        max_length=100, required=False, label="Excel sheet", help_text="Name of the sheet; empty for the first."
    )

    frame = None
    read_options = None

    def option_fields(self) -> list:
        return [self[name] for name in ("separator", "decimal", "encoding", "sheet")]

    def clean(self):
        cleaned = super().clean()
        upload = cleaned.get("file")
        if upload is None:
            return cleaned
        limit = settings.AUTOAD_WEB_MAX_UPLOAD_MB
        if upload.size > limit * 1024 * 1024:
            self.add_error("file", f"The file is larger than {limit} MB.")
            return cleaned
        options = {key: cleaned.get(key) for key in ("separator", "decimal", "encoding", "sheet")}
        try:
            self.frame, self.read_options = read_table(upload.read(), upload.name, options)
        except ReadError as error:
            self.add_error("file", str(error))
        finally:
            upload.seek(0)
        return cleaned

    def display_name(self) -> str:
        return self.cleaned_data.get("name") or self.cleaned_data["file"].name


class ConfigureForm(forms.Form):
    """Detector and AutoAD settings, plus one role per column (fields ``role_0``, ``role_1``, ...)."""

    detector = forms.ChoiceField(choices=[(key, spec["label"]) for key, spec in DETECTORS.items()])
    contamination = forms.FloatField(
        min_value=0.001,
        max_value=0.5,
        help_text="Share of the training rows assumed to be anomalies; sets the threshold for the labels.",
    )
    n_neighbors = forms.IntegerField(
        min_value=1,
        max_value=10_000,
        required=False,
        label="Neighbours",
        help_text="KNN and LOF only. Empty: PyOD's default (KNN 5, LOF 20).",
    )
    n_estimators = forms.IntegerField(
        min_value=10,
        max_value=5_000,
        required=False,
        label="Trees",
        help_text="IForest only. Empty: 100.",
    )
    drop_constant = forms.BooleanField(required=False, label="Drop constant columns and features")
    drop_id_columns = forms.BooleanField(required=False, label="Drop ID-like columns")
    max_binary_categories = forms.IntegerField(
        min_value=1,
        max_value=100_000,
        label="Binary code up to",
        help_text="Categorical columns with more distinct values are frequency-encoded.",
    )
    tukey_factor = forms.FloatField(min_value=0.1, max_value=100, label="Tukey factor", help_text="IQR multiple.")
    mad_threshold = forms.FloatField(
        min_value=0.1, max_value=100, label="MAD threshold", help_text="Limit for the modified z-score."
    )
    missing_indicator = forms.ChoiceField(
        choices=[("count", "One missing_count column"), ("columns", "One column per input column"), ("none", "None")],
        label="Missing values as",
    )
    scaler = forms.ChoiceField(
        choices=[("robust", "Robust"), ("standard", "Standard"), ("minmax", "Min-max"), ("none", "None")],
    )
    duplicate_tolerance = forms.FloatField(
        min_value=0,
        max_value=100,
        label="Tolerance",
        help_text="0 finds rows with identical values. A larger value also finds near duplicates up to this "
        "distance in AutoAD's scaled features. With the robust scaler, 0.05 allows one number to differ by 5 % "
        "of its interquartile range.",
    )

    ADVANCED = (
        "drop_constant",
        "drop_id_columns",
        "max_binary_categories",
        "tukey_factor",
        "mad_threshold",
        "missing_indicator",
        "scaler",
    )

    def __init__(self, *args, columns: list[str], current: dict | None = None, **kwargs):
        current = {**DEFAULT_SETTINGS, **(current or {})}
        roles = current.get("roles") or {}
        kwargs.setdefault("initial", {key: value for key, value in current.items() if key != "roles"})
        super().__init__(*args, **kwargs)
        self.columns = list(columns)
        for index, column in enumerate(self.columns):
            self.fields[f"role_{index}"] = forms.ChoiceField(
                choices=list(ROLES.items()), initial=roles.get(column, "auto"), label=column
            )

    def role_fields(self) -> list:
        return [self[f"role_{index}"] for index in range(len(self.columns))]

    def detector_fields(self) -> list:
        return [self[name] for name in ("detector", "contamination", "n_neighbors", "n_estimators")]

    def advanced_fields(self) -> list:
        return [self[name] for name in self.ADVANCED]

    def clean(self):
        cleaned = super().clean()
        roles = [cleaned.get(f"role_{index}") for index in range(len(self.columns))]
        if roles and all(role == "exclude" for role in roles):
            raise forms.ValidationError("At least one column must be used.")
        return cleaned

    def to_settings(self) -> dict:
        data = self.cleaned_data
        result = {key: data[key] for key in DEFAULT_SETTINGS if key != "roles"}
        result["roles"] = {
            column: data[f"role_{index}"]
            for index, column in enumerate(self.columns)
            if data[f"role_{index}"] != "auto"
        }
        return result
