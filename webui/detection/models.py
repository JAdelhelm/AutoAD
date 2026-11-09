"""An analysis is one fitted AutoAD model; a scoring is one table checked with it."""

import uuid
from pathlib import Path

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver


def upload_path(instance, filename: str) -> str:
    # Stored under a random name; the original name is kept in ``file_name``.
    return f"uploads/{uuid.uuid4().hex}{Path(filename).suffix.lower()}"


class Analysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    file_name = models.CharField(max_length=255)
    data_file = models.FileField(upload_to=upload_path)
    #: How the file was read (format, separator, decimal, encoding, sheet).
    read_options = models.JSONField(default=dict)
    n_rows = models.PositiveIntegerField()
    n_columns = models.PositiveIntegerField()
    #: Values of the configure form, see ``services.runner.DEFAULT_SETTINGS``.
    settings = models.JSONField(default=dict, blank=True)
    #: joblib file with the fitted AutoAD, the 2D projection and training statistics.
    model_file = models.FileField(upload_to="models/", blank=True)
    fitted_at = models.DateTimeField(null=True, blank=True)
    fit_seconds = models.FloatField(null=True, blank=True)
    #: Rows used for fitting; fewer than ``n_rows`` if a sample was used.
    train_rows = models.PositiveIntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "analyses"

    def __str__(self) -> str:
        return self.name

    @property
    def is_fitted(self) -> bool:
        return bool(self.model_file)

    @property
    def training_scoring(self):
        return next((s for s in self.scorings.all() if s.is_training), None)


class Scoring(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE, related_name="scorings")
    created_at = models.DateTimeField(auto_now_add=True)
    #: True for the scoring of the training file itself.
    is_training = models.BooleanField(default=False)
    name = models.CharField(max_length=200)
    file_name = models.CharField(max_length=255)
    #: The checked file; empty for the training data (stored on the analysis).
    data_file = models.FileField(upload_to=upload_path, blank=True)
    read_options = models.JSONField(default=dict)
    #: joblib file with the report, the findings, labels and 2D coordinates.
    result_file = models.FileField(upload_to="results/")
    n_rows = models.PositiveIntegerField()
    n_anomalies = models.PositiveIntegerField()
    threshold = models.FloatField()
    #: Row counts with findings: rows_extreme, rows_missing, rows_unseen.
    summary = models.JSONField(default=dict)

    class Meta:
        ordering = ["-is_training", "created_at"]

    def __str__(self) -> str:
        return f"{self.analysis.name}: {self.name}"

    @property
    def anomaly_share(self) -> float:
        return self.n_anomalies / self.n_rows if self.n_rows else 0.0


def _delete_files(instance, *field_names: str) -> None:
    for name in field_names:
        field = getattr(instance, name)
        if field:
            field.delete(save=False)


@receiver(post_delete, sender=Analysis)
def _delete_analysis_files(sender, instance, **kwargs):
    _delete_files(instance, "data_file", "model_file")


@receiver(post_delete, sender=Scoring)
def _delete_scoring_files(sender, instance, **kwargs):
    _delete_files(instance, "data_file", "result_file")
