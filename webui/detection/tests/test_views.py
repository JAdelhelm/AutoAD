import json
import os

from django.contrib.staticfiles import finders
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from detection.models import Analysis, Scoring
from detection.services.examples import EXAMPLES

from .helpers import ANOMALY_ROWS, DUPLICATE_ROWS, TemporaryMediaMixin, configure_data, csv_bytes, make_frame


class WorkflowTests(TemporaryMediaMixin, TestCase):
    def setUp(self):
        self.frame = make_frame()

    def upload(self, data: bytes, file_name="orders.csv", **extra):
        payload = {
            "file": SimpleUploadedFile(file_name, data),
            "name": "",
            "separator": "auto",
            "decimal": "auto",
            "encoding": "auto",
            "sheet": "",
            **extra,
        }
        return self.client.post(reverse("detection:index"), payload)

    def run_analysis(self, **overrides) -> Scoring:
        self.upload(csv_bytes(self.frame))
        analysis = Analysis.objects.get()
        response = self.client.post(
            reverse("detection:configure", args=[analysis.pk]), configure_data(self.frame.shape[1], **overrides)
        )
        scoring = analysis.scorings.get(is_training=True)
        self.assertRedirects(response, reverse("detection:scoring", args=[scoring.pk]))
        return scoring

    def test_index(self):
        response = self.client.get(reverse("detection:index"))
        self.assertContains(response, "New analysis")
        self.assertContains(response, "No analyses yet")

    def test_upload_leads_to_configure_page(self):
        response = self.upload(csv_bytes(self.frame, sep=";", decimal=","), name="Orders 2025")
        analysis = Analysis.objects.get()
        self.assertRedirects(response, reverse("detection:configure", args=[analysis.pk]))
        self.assertEqual(analysis.name, "Orders 2025")
        self.assertEqual((analysis.n_rows, analysis.n_columns), self.frame.shape)
        self.assertEqual(analysis.read_options["separator"], ";")

        page = self.client.get(reverse("detection:configure", args=[analysis.pk]))
        self.assertContains(page, "Channel")
        self.assertContains(page, "datetime")  # detected type of "Ordered"
        self.assertContains(page, 'name="role_4"')

    def test_unreadable_upload_shows_the_problem(self):
        response = self.upload(b"x", file_name="old.xls")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Save the file as .xlsx")
        self.assertFalse(Analysis.objects.exists())

    def test_full_workflow(self):
        scoring = self.run_analysis()
        page = self.client.get(reverse("detection:scoring", args=[scoring.pk]))
        self.assertContains(page, "Anomaly score per row")
        self.assertContains(page, 'id="chart-data"')
        figures = json.loads(page.context["figures"] and json.dumps(page.context["figures"]))
        self.assertEqual(sorted(figures), ["histogram", "importances", "map", "scores"])
        self.assertContains(page, "pd.read_csv(&#x27;orders.csv&#x27;")
        self.assertContains(page, "dropped: ID-like")
        self.assertContains(page, "Not compared: Order ID.")
        self.assertContains(page, "identical")
        self.assertEqual(page.context["duplicates"]["rows"], 2)

        duplicates = self.client.get(reverse("detection:scoring", args=[scoring.pk]), {"show": "duplicates"})
        numbers = [row["number"] for row in duplicates.context["table"]["rows"]]
        self.assertEqual(numbers, [row + 1 for row in DUPLICATE_ROWS])
        copy = self.client.get(reverse("detection:row", args=[scoring.pk, DUPLICATE_ROWS[1] + 1]))
        self.assertContains(copy, "Duplicate of row")

        anomalies = self.client.get(reverse("detection:scoring", args=[scoring.pk]), {"show": "anomalies"})
        self.assertEqual(anomalies.context["table"]["count"], scoring.n_anomalies)
        self.assertContains(anomalies, "cell-extreme")

        row = self.client.get(reverse("detection:row", args=[scoring.pk, ANOMALY_ROWS[0] + 1]))
        self.assertContains(row, f"Row {ANOMALY_ROWS[0] + 1}")
        self.assertContains(row, "High modified z-score")
        self.assertEqual(self.client.get(reverse("detection:row", args=[scoring.pk, 999])).status_code, 404)

        download = self.client.get(reverse("detection:download", args=[scoring.pk]))
        self.assertEqual(download["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn('filename="orders_autoad.csv"', download["Content-Disposition"])
        lines = download.content.decode("utf-8-sig").splitlines()
        header = lines[0].split(",")
        self.assertEqual(
            header[-6:],
            ["Unseen_Columns", "is_anomaly", "Duplicate_score", "Closest_row", "Duplicate_group", "is_duplicate"],
        )
        copy_line = lines[DUPLICATE_ROWS[1] + 1].split(",")
        self.assertEqual(copy_line[header.index("Closest_row")], str(DUPLICATE_ROWS[0] + 1))
        self.assertEqual(copy_line[header.index("is_duplicate")], "1")
        self.assertEqual(len(lines), len(self.frame) + 1)

        index = self.client.get(reverse("detection:index"))
        self.assertContains(index, "Isolation Forest")

    def test_check_new_data(self):
        scoring = self.run_analysis()
        new = make_frame(30, seed=1)
        new.loc[2, "Channel"] = "fax"
        response = self.client.post(
            reverse("detection:check", args=[scoring.analysis.pk]),
            {
                "file": SimpleUploadedFile("march.csv", csv_bytes(new)),
                "separator": "auto",
                "decimal": "auto",
                "encoding": "auto",
                "sheet": "",
                "name": "",
            },
        )
        check = Scoring.objects.get(is_training=False)
        self.assertRedirects(response, reverse("detection:scoring", args=[check.pk]))
        page = self.client.get(reverse("detection:scoring", args=[check.pk]))
        self.assertContains(page, "Rows with new categories")
        self.assertContains(page, "march.csv")

        wrong = self.client.post(
            reverse("detection:check", args=[scoring.analysis.pk]),
            {
                "file": SimpleUploadedFile("bad.csv", csv_bytes(new.drop(columns=["Amount"]))),
                "separator": "auto",
                "decimal": "auto",
                "encoding": "auto",
                "sheet": "",
                "name": "",
            },
        )
        self.assertContains(wrong, "Missing: [&#x27;Amount&#x27;]")

        path = check.result_file.path
        self.client.post(reverse("detection:delete_scoring", args=[check.pk]))
        self.assertFalse(Scoring.objects.filter(pk=check.pk).exists())
        self.assertFalse(os.path.exists(path))

    def test_configure_errors_are_shown(self):
        self.upload(csv_bytes(self.frame))
        analysis = Analysis.objects.get()
        all_ignored = configure_data(
            self.frame.shape[1], **{f"role_{i}": "exclude" for i in range(self.frame.shape[1])}
        )
        response = self.client.post(reverse("detection:configure", args=[analysis.pk]), all_ignored)
        self.assertContains(response, "At least one column must be used.")
        only_ids = configure_data(
            self.frame.shape[1], **{f"role_{i}": "exclude" for i in range(1, self.frame.shape[1])}
        )
        response = self.client.post(reverse("detection:configure", args=[analysis.pk]), only_ids)
        self.assertContains(response, "No usable columns left")

    def test_delete_analysis(self):
        scoring = self.run_analysis()
        analysis = Analysis.objects.get(pk=scoring.analysis_id)
        paths = [analysis.data_file.path, analysis.model_file.path, scoring.result_file.path]
        response = self.client.post(reverse("detection:delete_analysis", args=[analysis.pk]))
        self.assertRedirects(response, reverse("detection:index"))
        self.assertFalse(Analysis.objects.exists())
        self.assertEqual([os.path.exists(path) for path in paths], [False, False, False])
        self.assertEqual(self.client.get(reverse("detection:delete_analysis", args=[analysis.pk])).status_code, 405)

    def test_analysis_redirects_to_configure_until_it_ran(self):
        self.upload(csv_bytes(self.frame))
        analysis = Analysis.objects.get()
        response = self.client.get(reverse("detection:analysis", args=[analysis.pk]))
        self.assertRedirects(response, reverse("detection:configure", args=[analysis.pk]))


class ExampleTests(TemporaryMediaMixin, TestCase):
    def run_example(self, key: str) -> Scoring:
        response = self.client.post(reverse("detection:example", args=[key]))
        analysis = Analysis.objects.get()
        self.assertRedirects(response, reverse("detection:configure", args=[analysis.pk]))
        page = self.client.get(reverse("detection:configure", args=[analysis.pk]))
        # The example's presets are the initial values of the configure form.
        contamination = page.context["form"]["contamination"].value()
        self.client.post(
            reverse("detection:configure", args=[analysis.pk]),
            configure_data(analysis.n_columns, contamination=str(contamination)),
        )
        return analysis.scorings.get(is_training=True)

    def test_index_offers_the_examples(self):
        response = self.client.get(reverse("detection:index"))
        for example in EXAMPLES.values():
            self.assertContains(response, example.title)
            self.assertContains(response, reverse("detection:example", args=[example.key]))

    def test_examples_are_reproducible(self):
        for example in EXAMPLES.values():
            self.assertEqual(example.training_csv(), example.training_csv())
            self.assertNotEqual(example.training_csv(), example.check_csv())

    def test_every_example_runs_and_its_new_data_can_be_checked(self):
        for key, example in EXAMPLES.items():
            with self.subTest(example=key):
                Analysis.objects.all().delete()
                training = self.run_example(key)
                analysis = training.analysis
                self.assertEqual(analysis.name, f"Example: {example.title}")
                self.assertGreater(training.n_anomalies, 0)
                self.assertGreater(training.summary["rows_extreme"], 0)
                self.assertGreater(training.summary["rows_missing"], 0)

                page = self.client.get(reverse("detection:check", args=[analysis.pk]))
                self.assertContains(page, "Check the example data")
                response = self.client.post(reverse("detection:check", args=[analysis.pk]), {"example": "1"})
                check = analysis.scorings.get(is_training=False)
                self.assertRedirects(response, reverse("detection:scoring", args=[check.pk]))
                self.assertEqual(check.file_name, example.check_file_name)
                self.assertGreater(check.summary["rows_unseen"], 0)
                self.assertEqual(self.client.get(reverse("detection:download", args=[check.pk])).status_code, 200)

    def test_orders_example_contains_a_duplicate(self):
        self.assertEqual(self.run_example("orders").summary["duplicate_groups"], 1)

    def test_unknown_example_and_get_are_rejected(self):
        self.assertEqual(self.client.post(reverse("detection:example", args=["nope"])).status_code, 404)
        self.assertEqual(self.client.get(reverse("detection:example", args=["orders"])).status_code, 405)
        self.assertFalse(Analysis.objects.exists())

    def test_uploaded_analysis_gets_no_example_button(self):
        frame = make_frame()
        self.client.post(
            reverse("detection:index"),
            {
                "file": SimpleUploadedFile("orders.csv", csv_bytes(frame)),
                "separator": "auto",
                "decimal": "auto",
                "encoding": "auto",
            },
        )
        analysis = Analysis.objects.get()
        self.client.post(reverse("detection:configure", args=[analysis.pk]), configure_data(frame.shape[1]))
        page = self.client.get(reverse("detection:check", args=[analysis.pk]))
        self.assertNotContains(page, "Check the example data")
        response = self.client.post(reverse("detection:check", args=[analysis.pk]), {"example": "1"})
        self.assertEqual(response.status_code, 200)  # treated as an upload without a file
        self.assertFalse(analysis.scorings.filter(is_training=False).exists())


class AssetTests(TestCase):
    def test_plotly_is_served_from_the_installed_package(self):
        response = self.client.get(reverse("detection:plotly_js"))
        self.assertEqual(response["Content-Type"], "text/javascript")
        self.assertIn("max-age", response["Cache-Control"])
        self.assertIn(b"plotly", b"".join(response.streaming_content)[:2000].lower())

    def test_runserver_defaults_to_port_8090_and_serves_static_files(self):
        from django.contrib.staticfiles.management.commands.runserver import Command as StaticFilesRunserver
        from django.core.management import get_commands, load_command_class

        self.assertEqual(get_commands()["runserver"], "detection")
        command = load_command_class("detection", "runserver")
        self.assertEqual(command.default_port, os.environ.get("AUTOAD_WEB_PORT", "8090"))
        self.assertIsInstance(command, StaticFilesRunserver)

    def test_own_static_files_exist(self):
        self.assertIsNotNone(finders.find("detection/app.js"))
        self.assertIsNotNone(finders.find("detection/app.css"))
