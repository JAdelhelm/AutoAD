from __future__ import annotations

import logging
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.http import content_disposition_header
from django.views.decorators.cache import cache_control
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .forms import ConfigureForm, TableUploadForm
from .models import Analysis, Scoring
from .services import charts, tables
from .services.code import python_snippet
from .services.readers import ReadError, read_table
from .services.runner import DETECTORS, AnalysisError, check_new_data, fit_analysis, load_model, load_result

logger = logging.getLogger(__name__)


class StoredFileMissing(Exception):
    pass


def _read_analysis_file(analysis: Analysis):
    with analysis.data_file.open("rb") as handle:
        data = handle.read()
    frame, _ = read_table(data, analysis.file_name, analysis.read_options)
    return frame


def _load(scoring: Scoring) -> tuple[dict, dict]:
    try:
        return load_model(scoring.analysis), load_result(scoring)
    except (OSError, EOFError, ValueError) as error:
        logger.exception("Cannot load stored results")
        raise StoredFileMissing(str(error)) from error


def _detector_label(analysis: Analysis) -> str:
    detector = (analysis.settings or {}).get("detector")
    return DETECTORS[detector]["label"] if detector in DETECTORS else ""


@cache
def _plotly_js() -> tuple[Path | None, str]:
    """Path of plotly.min.js in the installed plotly package, and the package version."""
    spec = find_spec("plotly")
    if spec is None or not spec.submodule_search_locations:
        return None, ""
    path = Path(next(iter(spec.submodule_search_locations))) / "package_data" / "plotly.min.js"
    try:
        package_version = version("plotly")
    except PackageNotFoundError:
        package_version = ""
    return (path if path.is_file() else None), package_version


@require_GET
@cache_control(max_age=30 * 24 * 3600, public=True)
def plotly_js(request):
    # Served from the plotly package so the charts work offline. The URL carries the
    # package version, so a long cache lifetime is safe.
    path, _ = _plotly_js()
    if path is None:
        raise Http404("plotly is not installed: pip install plotly")
    return FileResponse(path.open("rb"), content_type="text/javascript")


@require_http_methods(["GET", "POST"])
def index(request):
    if request.method == "POST":
        form = TableUploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload = form.cleaned_data["file"]
            analysis = Analysis(
                name=form.display_name(),
                file_name=upload.name,
                read_options=form.read_options,
                n_rows=form.frame.shape[0],
                n_columns=form.frame.shape[1],
            )
            analysis.data_file.save(upload.name, upload, save=False)
            analysis.save()
            return redirect("detection:configure", pk=analysis.pk)
    else:
        form = TableUploadForm()
    analyses = [
        {"analysis": analysis, "detector": _detector_label(analysis), "training": analysis.training_scoring}
        for analysis in Analysis.objects.prefetch_related("scorings")
    ]
    return render(
        request,
        "detection/index.html",
        {"form": form, "analyses": analyses, "max_upload_mb": settings.AUTOAD_WEB_MAX_UPLOAD_MB},
    )


@require_http_methods(["GET", "POST"])
def configure(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk)
    try:
        frame = _read_analysis_file(analysis)
    except (ReadError, OSError) as error:
        messages.error(request, f"The stored file of “{analysis.name}” cannot be read: {error}")
        return redirect("detection:index")

    columns = list(frame.columns)
    if request.method == "POST":
        form = ConfigureForm(request.POST, columns=columns, current=analysis.settings)
        if form.is_valid():
            try:
                scoring = fit_analysis(analysis, frame, form.to_settings())
            except AnalysisError as error:
                form.add_error(None, str(error))
            else:
                return redirect("detection:scoring", pk=scoring.pk)
    else:
        form = ConfigureForm(columns=columns, current=analysis.settings)

    preview = frame.head(10)
    return render(
        request,
        "detection/configure.html",
        {
            "analysis": analysis,
            "form": form,
            "columns": list(zip(tables.column_overview(frame), form.role_fields(), strict=True)),
            "preview_columns": columns,
            "preview_rows": [[tables.format_value(value) for value in row] for row in preview.itertuples(index=False)],
            "max_train_rows": settings.AUTOAD_WEB_MAX_TRAIN_ROWS,
            "has_checks": analysis.scorings.filter(is_training=False).exists(),
        },
    )


def analysis_detail(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk)
    training = analysis.training_scoring
    if training is None:
        return redirect("detection:configure", pk=analysis.pk)
    return redirect("detection:scoring", pk=training.pk)


def scoring_detail(request, pk):
    scoring = get_object_or_404(Scoring.objects.select_related("analysis"), pk=pk)
    analysis = scoring.analysis
    try:
        model, result = _load(scoring)
    except StoredFileMissing:
        messages.error(request, "The stored results cannot be loaded. Run the analysis again.")
        return redirect("detection:configure", pk=analysis.pk)

    show = request.GET.get("show")
    show = show if show in tables.SHOW_CHOICES else "all"
    order = request.GET.get("order")
    order = order if order in tables.ORDER_CHOICES else "score"
    table = tables.row_table(
        result, show=show, order=order, page=request.GET.get("page"), per_page=settings.AUTOAD_WEB_ROWS_PER_PAGE
    )

    figures = charts.dashboard(result, model, settings.AUTOAD_WEB_MAX_PLOT_POINTS)
    ad = model["autoad"]
    notes = list(model["notes"])
    if analysis.train_rows and analysis.train_rows < analysis.n_rows:
        notes.append(
            f"The model was fitted on a random sample of {analysis.train_rows:,} of {analysis.n_rows:,} rows. "
            "All rows were scored."
        )
    row_url = reverse("detection:row", args=[scoring.pk, 999_999_999]).replace("999999999", charts.ROW_PLACEHOLDER)
    return render(
        request,
        "detection/scoring.html",
        {
            "analysis": analysis,
            "scoring": scoring,
            "scorings": analysis.scorings.all(),
            "detector": _detector_label(analysis),
            "notes": notes,
            "figures": {key: {"data": f["data"], "layout": f["layout"]} for key, f in figures.items()},
            "sampled": any(f.get("sampled") for f in figures.values()),
            "max_plot_points": settings.AUTOAD_WEB_MAX_PLOT_POINTS,
            "plotly_version": _plotly_js()[1],
            "row_url": row_url,
            "column_findings": tables.column_findings(result, ad),
            "duplicates": tables.duplicate_groups(result),
            "duplicate_tolerance": model.get("duplicate_tolerance", 0.0),
            "ignored_for_duplicates": tables.compared_columns(ad, result["columns"])[1],
            "table": table,
            "show": show,
            "order": order,
            "show_choices": tables.SHOW_CHOICES,
            "order_choices": tables.ORDER_CHOICES,
            "findings_labels": tables.FINDINGS,
            "column_types": ad.column_types_,
            "dropped_columns": ad.dropped_columns_,
            "excluded_columns": ad.excluded_columns_,
            "dropped_features": ad.dropped_features_,
            "features": list(ad.get_feature_names_out()),
            "code": python_snippet(analysis, model),
        },
    )


def row_detail(request, pk, number):
    scoring = get_object_or_404(Scoring.objects.select_related("analysis"), pk=pk)
    if not 1 <= number <= scoring.n_rows:
        raise Http404("No such row.")
    try:
        model, result = _load(scoring)
    except StoredFileMissing:
        messages.error(request, "The stored results cannot be loaded. Run the analysis again.")
        return redirect("detection:configure", pk=scoring.analysis.pk)
    return render(
        request,
        "detection/row.html",
        {
            "analysis": scoring.analysis,
            "scoring": scoring,
            "row": tables.explain_row(result, model, number - 1),
        },
    )


def download_report(request, pk):
    scoring = get_object_or_404(Scoring.objects.select_related("analysis"), pk=pk)
    try:
        _, result = _load(scoring)
    except StoredFileMissing as error:
        raise Http404("The stored results cannot be loaded.") from error
    report = result["report"].copy()
    report["is_anomaly"] = result["labels"].astype(int)
    duplicates = tables.duplicate_info(result)
    report["Duplicate_score"] = duplicates["score"]
    # Row numbers as on the pages: 1 is the first data row of the file.
    report["Closest_row"] = duplicates["closest"] + 1
    report["Duplicate_group"] = result["duplicates"]["Duplicate_group"].to_numpy()
    report["is_duplicate"] = duplicates["is_duplicate"].astype(int)
    options = scoring.analysis.read_options if scoring.is_training else scoring.read_options
    # Write the CSV the way the input was written, so it opens the same way (e.g. German Excel).
    sep, decimal = ",", "."
    if options.get("format") == "csv":
        sep, decimal = options.get("separator", ","), options.get("decimal", ".")
    content = report.to_csv(index=False, sep=sep, decimal=decimal)
    response = HttpResponse(content.encode("utf-8-sig"), content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = content_disposition_header(
        as_attachment=True, filename=f"{Path(scoring.file_name).stem}_autoad.csv"
    )
    return response


@require_http_methods(["GET", "POST"])
def check(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk)
    if not analysis.is_fitted:
        return redirect("detection:configure", pk=analysis.pk)
    if request.method == "POST":
        form = TableUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                scoring = check_new_data(
                    analysis,
                    form.frame,
                    upload=form.cleaned_data["file"],
                    name=form.display_name(),
                    read_options=form.read_options,
                )
            except AnalysisError as error:
                form.add_error("file", str(error))
            else:
                return redirect("detection:scoring", pk=scoring.pk)
    else:
        form = TableUploadForm()
    return render(
        request,
        "detection/check.html",
        {"analysis": analysis, "form": form, "max_upload_mb": settings.AUTOAD_WEB_MAX_UPLOAD_MB},
    )


@require_POST
def delete_analysis(request, pk):
    analysis = get_object_or_404(Analysis, pk=pk)
    analysis.delete()
    messages.success(request, f"Deleted “{analysis.name}”.")
    return redirect("detection:index")


@require_POST
def delete_scoring(request, pk):
    scoring = get_object_or_404(Scoring.objects.select_related("analysis"), pk=pk)
    if scoring.is_training:
        raise Http404("The training data can only be deleted with the whole analysis.")
    analysis = scoring.analysis
    scoring.delete()
    messages.success(request, f"Deleted the check “{scoring.name}”.")
    return redirect("detection:analysis", pk=analysis.pk)
