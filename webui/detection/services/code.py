"""Python code that repeats an analysis with the AutoAD library."""

from __future__ import annotations

from autoad import AutoAD

from .readers import pandas_read_call
from .runner import DETECTORS, autoad_arguments


def python_snippet(analysis, model: dict) -> str:
    settings = analysis.settings
    spec = DETECTORS[settings["detector"]]
    detector_args = ", ".join(f"{key}={value!r}" for key, value in model["detector_kwargs"].items())
    defaults = AutoAD().get_params()
    arguments = [f"    detector={spec['class']}({detector_args}),"]
    for key, value in autoad_arguments(settings).items():
        if value != defaults[key]:
            arguments.append(f"    {key}={value!r},")

    if spec["module"] == "autoad":
        imports = ["", f"from autoad import AutoAD, {spec['class']}"]
    else:
        imports = [f"from {spec['module']} import {spec['class']}", "", "from autoad import AutoAD"]
    lines = [
        "import pandas as pd",
        *imports,
        "",
        f"X = {pandas_read_call(analysis.file_name, analysis.read_options)}",
    ]
    if analysis.train_rows and analysis.train_rows < analysis.n_rows:
        lines.append(f"X_train = X.sample(n={analysis.train_rows}, random_state=0).sort_index()")
        train = "X_train"
    else:
        train = "X"
    lines += [
        "",
        "ad = AutoAD(",
        *arguments,
        f").fit({train})",
        "",
        "report = ad.report(X)  # X plus AD_score and the findings per row",
        'report["is_anomaly"] = ad.predict(X)',
        'report.sort_values("AD_score", ascending=False).head(20)',
        "",
        "# Rows that match another row: KNN with 2 neighbours, a low score marks a duplicate",
    ]
    tolerance = model.get("duplicate_tolerance") or 0.0
    lines.append(
        f"duplicates = ad.duplicates(X, tolerance={tolerance!r})" if tolerance else "duplicates = ad.duplicates(X)"
    )
    lines.append('duplicates[duplicates["is_duplicate"]].sort_values("Duplicate_group")')
    return "\n".join(lines)
