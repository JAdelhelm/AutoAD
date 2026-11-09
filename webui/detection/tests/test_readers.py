import io

import pandas as pd
from django.test import SimpleTestCase

from detection.services.readers import ReadError, pandas_read_call, read_table, sniff_separator


class SniffSeparatorTests(SimpleTestCase):
    def test_comma(self):
        self.assertEqual(sniff_separator("a,b\n1,2\n3,4\n"), ",")

    def test_semicolon_with_decimal_comma(self):
        # Every line also contains exactly one comma; the header decides.
        self.assertEqual(sniff_separator("Name;Salary\nAnna;3,5\nBen;4,25\n"), ";")

    def test_tab_and_pipe(self):
        self.assertEqual(sniff_separator("a\tb\tc\n1\t2\t3\n"), "\t")
        self.assertEqual(sniff_separator("a|b\n1|2\n"), "|")

    def test_quoted_commas_do_not_count(self):
        self.assertEqual(sniff_separator('"Doe, Jane";age\n"Roe, Rick";30\n'), ";")

    def test_single_column_falls_back_to_comma(self):
        self.assertEqual(sniff_separator("value\n1\n2\n"), ",")


class ReadTextTests(SimpleTestCase):
    def test_comma_csv(self):
        frame, used = read_table(b"a,b\n1,x\n2,y\n", "data.csv")
        self.assertEqual(list(frame.columns), ["a", "b"])
        self.assertEqual(frame["a"].tolist(), [1, 2])
        self.assertEqual(used, {"separator": ",", "decimal": ".", "encoding": "utf-8-sig", "format": "csv"})

    def test_german_excel_csv(self):
        data = "Name;Gehalt\nAnna;3500,5\nJürgen;4200,25\n".encode("cp1252")
        frame, used = read_table(data, "gehalt.csv")
        self.assertEqual(frame["Gehalt"].tolist(), [3500.5, 4200.25])
        self.assertEqual(frame["Name"].tolist(), ["Anna", "Jürgen"])
        self.assertEqual((used["separator"], used["decimal"], used["encoding"]), (";", ",", "cp1252"))

    def test_explicit_options_win(self):
        frame, used = read_table(b"a;b\n1.5;2\n", "data.txt", {"separator": "semicolon", "decimal": "point"})
        self.assertEqual(frame["a"].tolist(), [1.5])
        self.assertEqual(used["decimal"], ".")

    def test_stored_options_can_be_reused(self):
        _, used = read_table(b"a;b\n1,5;2\n", "data.csv")
        frame, _ = read_table(b"a;b\n1,5;2\n", "data.csv", used)
        self.assertEqual(frame["a"].tolist(), [1.5])

    def test_tsv(self):
        frame, used = read_table(b"a\tb\n1\t2\n", "data.tsv")
        self.assertEqual(used["separator"], "\t")
        self.assertEqual(frame.shape, (1, 2))

    def test_utf8_bom_is_removed(self):
        frame, _ = read_table("﻿a,b\n1,2\n".encode(), "data.csv")
        self.assertEqual(list(frame.columns), ["a", "b"])

    def test_wrong_explicit_encoding(self):
        with self.assertRaisesMessage(ReadError, "cannot be decoded as utf-8"):
            read_table("a\nü\n".encode("cp1252"), "data.csv", {"encoding": "utf-8"})

    def test_same_separator_and_decimal(self):
        with self.assertRaisesMessage(ReadError, "must be different"):
            read_table(b"a,b\n1,2\n", "data.csv", {"separator": "comma", "decimal": "comma"})

    def test_empty_and_header_only(self):
        with self.assertRaisesMessage(ReadError, "empty"):
            read_table(b"", "data.csv")
        with self.assertRaisesMessage(ReadError, "no data rows"):
            read_table(b"a,b\n", "data.csv")


class ReadOtherFormatsTests(SimpleTestCase):
    def test_excel_first_and_named_sheet(self):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="First", index=False)
            pd.DataFrame({"b": [3, 4, 5]}).to_excel(writer, sheet_name="Second", index=False)
        data = buffer.getvalue()
        frame, used = read_table(data, "book.xlsx")
        self.assertEqual(list(frame.columns), ["a"])
        self.assertEqual(used, {"sheet": "", "format": "excel"})
        frame, _ = read_table(data, "book.xlsx", {"sheet": "Second"})
        self.assertEqual(frame["b"].tolist(), [3, 4, 5])
        with self.assertRaisesMessage(ReadError, "cannot be read"):
            read_table(data, "book.xlsx", {"sheet": "Missing"})

    def test_broken_excel(self):
        with self.assertRaisesMessage(ReadError, "not a valid .xlsx"):
            read_table(b"not a zip file", "book.xlsx")

    def test_json_records_and_lines(self):
        frame, used = read_table(b'[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]', "data.json")
        self.assertEqual(frame["a"].tolist(), [1, 2])
        self.assertEqual(used, {"format": "json"})
        frame, used = read_table(b'{"a": 1}\n{"a": 2}\n', "data.jsonl")
        self.assertEqual(frame["a"].tolist(), [1, 2])
        self.assertEqual(used["format"], "jsonl")

    def test_unsupported_files(self):
        with self.assertRaisesMessage(ReadError, "Save the file as .xlsx"):
            read_table(b"x", "old.xls")
        with self.assertRaisesMessage(ReadError, "Unsupported file type '.pdf'"):
            read_table(b"x", "report.pdf")

    def test_column_names_become_strings(self):
        frame, _ = read_table(b'[{"1": 1, "2": 2}]', "data.json")
        self.assertTrue(all(isinstance(col, str) for col in frame.columns))


class PandasReadCallTests(SimpleTestCase):
    def test_csv_call_repeats_detected_options(self):
        call = pandas_read_call("data.csv", {"format": "csv", "separator": ";", "decimal": ",", "encoding": "cp1252"})
        self.assertEqual(
            call, "pd.read_csv('data.csv', sep=';', decimal=',', encoding='cp1252', skipinitialspace=True)"
        )

    def test_other_formats(self):
        self.assertEqual(
            pandas_read_call("b.xlsx", {"format": "excel", "sheet": "S"}), "pd.read_excel('b.xlsx', sheet_name='S')"
        )
        self.assertEqual(pandas_read_call("d.jsonl", {"format": "jsonl"}), "pd.read_json('d.jsonl', lines=True)")
        self.assertEqual(pandas_read_call("d.parquet", {"format": "parquet"}), "pd.read_parquet('d.parquet')")
