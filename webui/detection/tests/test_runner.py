import os

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from detection.models import Analysis, Scoring
from detection.services import charts, tables
from detection.services.code import python_snippet
from detection.services.runner import (
    DEFAULT_SETTINGS,
    AnalysisError,
    check_new_data,
    detector_arguments,
    fit_analysis,
    load_model,
    load_result,
)

from .helpers import ANOMALY_ROWS, DUPLICATE_ROWS, TemporaryMediaMixin, csv_bytes, make_frame


def create_analysis(frame, name="orders") -> Analysis:
    analysis = Analysis(
        name=name,
        file_name=f"{name}.csv",
        read_options={"format": "csv", "separator": ",", "decimal": ".", "encoding": "utf-8-sig"},
        n_rows=len(frame),
        n_columns=frame.shape[1],
    )
    analysis.data_file.save(f"{name}.csv", SimpleUploadedFile(f"{name}.csv", csv_bytes(frame)), save=False)
    analysis.save()
    return analysis


def settings_with(**overrides) -> dict:
    return {**DEFAULT_SETTINGS, "contamination": 0.05, **overrides}


class FitAnalysisTests(TemporaryMediaMixin, TestCase):
    def setUp(self):
        self.frame = make_frame()
        self.analysis = create_analysis(self.frame)

    def test_injected_anomalies_are_found(self):
        scoring = fit_analysis(self.analysis, self.frame, settings_with())
        self.assertTrue(scoring.is_training)
        self.assertEqual(scoring.n_rows, len(self.frame))
        result = load_result(scoring)
        for row in ANOMALY_ROWS:
            self.assertTrue(result["labels"][row], f"row {row} should be an anomaly")
        top = np.argsort(-result["report"]["AD_score"].to_numpy())[:2]
        self.assertEqual(sorted(top), list(ANOMALY_ROWS))
        self.assertGreaterEqual(scoring.summary["rows_extreme"], 2)
        self.assertEqual(scoring.summary["rows_unseen"], 0)

    def test_ensemble_finds_the_injected_anomalies(self):
        scoring = fit_analysis(self.analysis, self.frame, settings_with(detector="ensemble"))
        result = load_result(scoring)
        top = np.argsort(-result["report"]["AD_score"].to_numpy())[:2]
        self.assertEqual(sorted(top), list(ANOMALY_ROWS))
        code = python_snippet(self.analysis, load_model(self.analysis))
        compile(code, "snippet", "exec")
        self.assertIn("from autoad import AutoAD, ScoreEnsemble", code)
        self.assertIn("detector=ScoreEnsemble(contamination=0.05, random_state=0)", code)

    def test_duplicates_are_found_despite_different_ids(self):
        scoring = fit_analysis(self.analysis, self.frame, settings_with())
        self.assertEqual((scoring.summary["rows_duplicate"], scoring.summary["duplicate_groups"]), (2, 1))
        duplicates = load_result(scoring)["duplicates"]
        self.assertEqual(duplicates.index[duplicates["is_duplicate"]].tolist(), list(DUPLICATE_ROWS))

    def test_duplicate_tolerance_is_stored_with_the_model(self):
        fit_analysis(self.analysis, self.frame, settings_with(duplicate_tolerance=0.5))
        model = load_model(self.analysis)
        self.assertEqual(model["duplicate_tolerance"], 0.5)
        self.assertIn("ad.duplicates(X, tolerance=0.5)", python_snippet(self.analysis, model))

    def test_model_contents(self):
        fit_analysis(self.analysis, self.frame, settings_with())
        self.analysis.refresh_from_db()
        model = load_model(self.analysis)
        ad = model["autoad"]
        self.assertEqual(ad.dropped_columns_, {"Order ID": "id"})
        self.assertEqual(ad.column_types_["Ordered"], "datetime")
        self.assertIn("Amount", model["statistics"]["numeric"])
        self.assertIn("Channel", model["statistics"]["categories"])
        self.assertIsNotNone(model["projection"])
        self.assertEqual(self.analysis.train_rows, len(self.frame))

    def test_roles_are_passed_to_autoad(self):
        settings = settings_with(roles={"Channel": "exclude", "Quantity": "numeric"})
        fit_analysis(self.analysis, self.frame, settings)
        ad = load_model(self.analysis)["autoad"]
        self.assertEqual(ad.excluded_columns_, ["Channel"])
        self.assertEqual(ad.numerical_columns, ["Quantity"])

    def test_running_again_replaces_results_and_files(self):
        first = fit_analysis(self.analysis, self.frame, settings_with())
        old_result = first.result_file.path
        check = check_new_data(
            self.analysis,
            self.frame.head(20),
            upload=SimpleUploadedFile("new.csv", csv_bytes(self.frame.head(20))),
            name="new",
            read_options={},
        )
        old_upload = check.data_file.path
        second = fit_analysis(self.analysis, self.frame, settings_with(detector="ecod"))
        self.assertEqual(list(self.analysis.scorings.all()), [second])
        self.assertFalse(os.path.exists(old_result))
        self.assertFalse(os.path.exists(old_upload))

    @override_settings(AUTOAD_WEB_MAX_TRAIN_ROWS=50)
    def test_large_tables_are_fitted_on_a_sample(self):
        scoring = fit_analysis(self.analysis, self.frame, settings_with())
        self.analysis.refresh_from_db()
        self.assertEqual(self.analysis.train_rows, 50)
        self.assertEqual(scoring.n_rows, len(self.frame))
        self.assertIn("X.sample(n=50, random_state=0)", python_snippet(self.analysis, load_model(self.analysis)))

    def test_user_errors_become_analysis_errors(self):
        with self.assertRaisesMessage(AnalysisError, "No columns left"):
            roles = dict.fromkeys(self.frame.columns, "exclude")
            fit_analysis(self.analysis, self.frame, settings_with(roles=roles))
        self.assertFalse(Scoring.objects.exists())

    def test_deleting_the_analysis_removes_its_files(self):
        scoring = fit_analysis(self.analysis, self.frame, settings_with())
        paths = [self.analysis.data_file.path, self.analysis.model_file.path, scoring.result_file.path]
        self.analysis.delete()
        self.assertEqual([os.path.exists(path) for path in paths], [False, False, False])


class CheckNewDataTests(TemporaryMediaMixin, TestCase):
    def setUp(self):
        self.frame = make_frame()
        self.analysis = create_analysis(self.frame)
        fit_analysis(self.analysis, self.frame, settings_with())

    def test_new_data_with_unseen_category_and_missing_value(self):
        new = make_frame(30, seed=1)
        new.loc[3, "Channel"] = "fax"
        new.loc[5, "Amount"] = np.nan
        new = new[list(reversed(new.columns))]  # column order does not matter
        scoring = check_new_data(
            self.analysis, new, upload=SimpleUploadedFile("new.csv", csv_bytes(new)), name="March", read_options={}
        )
        self.assertFalse(scoring.is_training)
        self.assertEqual(scoring.summary["rows_unseen"], 1)
        self.assertEqual(scoring.summary["rows_missing"], 1)
        result = load_result(scoring)
        self.assertTrue(result["findings"]["Unseen"].loc[3, "Channel"])

    def test_missing_columns_are_reported(self):
        new = self.frame.drop(columns=["Amount"])
        with self.assertRaisesMessage(AnalysisError, "Missing: ['Amount']"):
            check_new_data(
                self.analysis, new, upload=SimpleUploadedFile("new.csv", csv_bytes(new)), name="x", read_options={}
            )


class DetectorArgumentTests(TestCase):
    def test_neighbours_are_limited_by_the_training_rows(self):
        kwargs, notes = detector_arguments(settings_with(detector="lof"), n_train=10)
        self.assertEqual(kwargs["n_neighbors"], 9)
        self.assertIn("reduced from 20 to 9", notes[0])
        kwargs, notes = detector_arguments(settings_with(detector="knn", n_neighbors=3), n_train=10)
        self.assertEqual((kwargs["n_neighbors"], notes), (3, []))

    def test_iforest_is_reproducible(self):
        kwargs, _ = detector_arguments(settings_with(n_estimators=50), n_train=100)
        self.assertEqual(kwargs, {"contamination": 0.05, "n_estimators": 50, "random_state": 0})

    def test_ensemble_is_reproducible_and_ignores_the_tree_count(self):
        kwargs, notes = detector_arguments(settings_with(detector="ensemble", n_estimators=50), n_train=100)
        self.assertEqual((kwargs, notes), ({"contamination": 0.05, "random_state": 0}, []))


class PresentationTests(TemporaryMediaMixin, TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.frame = make_frame()

    def setUp(self):
        self.analysis = create_analysis(self.frame)
        self.scoring = fit_analysis(self.analysis, self.frame, settings_with())
        self.result = load_result(self.scoring)
        self.model = load_model(self.analysis)

    def test_dashboard_figures(self):
        figures = charts.dashboard(self.result, self.model, max_points=20_000)
        self.assertEqual(sorted(figures), ["histogram", "importances", "map", "scores"])
        normal, anomaly = figures["scores"]["data"]
        self.assertEqual(len(normal["x"]) + len(anomaly["x"]), len(self.frame))
        self.assertIn(ANOMALY_ROWS[0] + 1, anomaly["x"])
        histogram = figures["histogram"]["data"]
        self.assertEqual(sum(histogram[1]["y"]), self.scoring.n_anomalies)

    def test_map_separates_the_anomalies(self):
        coords = np.asarray(self.result["coords"])
        distance = np.hypot(coords[:, 0], coords[:, 1])
        normal = np.delete(distance, ANOMALY_ROWS)
        for row in ANOMALY_ROWS:
            self.assertGreater(distance[row], np.quantile(normal, 0.95))

    def test_scores_share_one_precision(self):
        self.assertEqual(tables.score_decimals(np.array([-0.2, 0.125])), 4)
        self.assertEqual(tables.score_decimals(np.array([1.0, 8.5])), 3)
        self.assertEqual(tables.score_decimals(np.array([0.0])), 4)
        self.assertEqual(tables.format_score(2.24e-17, 4), "0.0000")
        self.assertEqual(tables.format_score(-7e-6, 4), "0.0000")
        self.assertEqual(tables.format_score(-0.0123, 4), "-0.0123")

    def test_plots_keep_every_anomaly_when_sampling(self):
        labels = np.zeros(1000, dtype=bool)
        labels[[5, 500, 999]] = True
        positions, sampled = charts.plot_positions(labels, max_points=100)
        self.assertTrue(sampled)
        self.assertEqual(len(positions), 100)
        self.assertTrue({5, 500, 999} <= set(positions.tolist()))

    def test_row_table_filters_and_marks_cells(self):
        table = tables.row_table(self.result, show="anomalies", order="score", page=1, per_page=50)
        self.assertEqual(table["count"], self.scoring.n_anomalies)
        first = table["rows"][0]
        self.assertEqual(first["rank"], 1)
        self.assertIn(first["number"] - 1, ANOMALY_ROWS)
        amount = first["cells"][table["columns"].index("Amount")]
        self.assertIn("cell-extreme", amount.css)

        in_file_order = tables.row_table(self.result, show="all", order="row", page=2, per_page=50)
        self.assertEqual(in_file_order["rows"][0]["number"], 51)

    def test_column_findings(self):
        entries = {entry["name"]: entry for entry in tables.column_findings(self.result, self.model["autoad"])}
        self.assertEqual(entries["Order ID"]["status"], "dropped: ID-like")
        self.assertGreaterEqual(entries["Amount"]["counts"]["MAD"], 2)
        self.assertIsNone(entries["Channel"]["counts"]["MAD"])

    def test_explain_row(self):
        row = tables.explain_row(self.result, self.model, ANOMALY_ROWS[0])
        self.assertEqual(row["number"], ANOMALY_ROWS[0] + 1)
        self.assertTrue(row["is_anomaly"])
        first = row["cells"][0]
        self.assertTrue(first["findings"])
        amount = next(cell for cell in row["cells"] if cell["name"] == "Amount")
        # The strip ends at the 99th percentile, so the outlier does not set the scale.
        self.assertEqual(amount["strip"]["outside"], "above")
        self.assertEqual(amount["strip"]["value"], 100.0)
        self.assertIn("98 % between", amount["context"])
        channel = next(cell for cell in row["cells"] if cell["name"] == "Channel")
        self.assertIn("of the training rows", channel["context"])

    def test_duplicate_groups_and_filter(self):
        first, second = (row + 1 for row in DUPLICATE_ROWS)
        groups = tables.duplicate_groups(self.result)
        self.assertEqual((groups["count"], groups["rows"]), (1, 2))
        self.assertEqual(groups["groups"][0]["rows"], [first, second])
        self.assertTrue(groups["groups"][0]["identical"])
        self.assertEqual(groups["groups"][0]["score"], "0")

        table = tables.row_table(self.result, show="duplicates", order="score", page=1, per_page=50)
        self.assertEqual([row["number"] for row in table["rows"]], [first, second])
        self.assertEqual(table["rows"][0]["duplicate"], {"group": 1, "closest": second})

        row = tables.explain_row(self.result, self.model, DUPLICATE_ROWS[1])
        self.assertEqual(row["duplicate"]["closest"], first)
        self.assertTrue(row["duplicate"]["is_duplicate"])
        other = tables.explain_row(self.result, self.model, 0)
        self.assertFalse(other["duplicate"]["is_duplicate"])

        compared, ignored = tables.compared_columns(self.model["autoad"], self.result["columns"])
        self.assertEqual(ignored, ["Order ID"])

    def test_python_snippet_is_valid_code(self):
        code = python_snippet(self.analysis, self.model)
        compile(code, "snippet", "exec")
        self.assertIn("IForest(contamination=0.05, random_state=0)", code)
        self.assertIn("pd.read_csv('orders.csv', skipinitialspace=True)", code)
        self.assertIn("duplicates = ad.duplicates(X)", code)
