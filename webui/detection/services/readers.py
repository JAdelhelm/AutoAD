"""Read an uploaded table into a DataFrame.

Supported: CSV/TSV/TXT, Excel (.xlsx, .xlsm), JSON (records or one object per line) and
Parquet (needs ``pyarrow``). For text files the separator, decimal mark and encoding are
detected unless given; the values actually used are returned, so the same file can be
read again later and the generated Python code matches.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

SEPARATORS = {"auto": None, "comma": ",", "semicolon": ";", "tab": "\t", "pipe": "|"}
DECIMALS = {"auto": None, "point": ".", "comma": ","}
ENCODINGS = ("auto", "utf-8", "cp1252", "latin-1")
#: Tried in this order when the encoding is "auto". latin-1 accepts any byte sequence.
_AUTO_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

TEXT_SUFFIXES = {".csv", ".tsv", ".txt"}
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}
PARQUET_SUFFIXES = {".parquet"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | EXCEL_SUFFIXES | JSON_SUFFIXES | PARQUET_SUFFIXES


class ReadError(Exception):
    """The file cannot be read; the message is shown to the user."""


def file_format(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return "csv"
    if suffix in EXCEL_SUFFIXES:
        return "excel"
    if suffix in JSON_SUFFIXES:
        return "jsonl" if suffix in (".jsonl", ".ndjson") else "json"
    if suffix in PARQUET_SUFFIXES:
        return "parquet"
    supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
    if suffix == ".xls":
        raise ReadError(f"Old Excel files (.xls) are not supported. Save the file as .xlsx. Supported: {supported}.")
    raise ReadError(f"Unsupported file type {suffix or '(none)'!r}. Supported: {supported}.")


def read_table(data: bytes, file_name: str, options: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Return the table and the read options that were used.

    ``options`` may contain ``separator`` and ``decimal`` (keys of :data:`SEPARATORS`
    and :data:`DECIMALS`, or the characters themselves), ``encoding`` and ``sheet``.
    """
    options = dict(options or {})
    kind = file_format(file_name)
    if not data:
        raise ReadError("The file is empty.")
    if kind == "csv":
        frame, used = _read_text(data, file_name, options)
    elif kind == "excel":
        frame, used = _read_excel(data, options)
    elif kind in ("json", "jsonl"):
        frame, used = _read_json(data, lines=kind == "jsonl")
    else:
        frame, used = _read_parquet(data)
    used["format"] = kind
    return _clean(frame), used


def _clean(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] == 0:
        raise ReadError("No columns found in the file.")
    if frame.shape[0] == 0:
        raise ReadError("The file has a header but no data rows.")
    frame = frame.set_axis([str(col) for col in frame.columns], axis=1)
    duplicates = sorted({col for col in frame.columns if list(frame.columns).count(col) > 1})
    if duplicates:
        raise ReadError(f"Column names must be unique. Repeated: {', '.join(duplicates)}.")
    return frame.reset_index(drop=True)


def _choice(value, mapping: dict):
    """Map a form choice ("semicolon") or a literal character (";") to the character."""
    if value in (None, "", "auto"):
        return None
    if value in mapping:
        return mapping[value]
    if value in mapping.values():
        return value
    raise ReadError(f"Unknown option {value!r}.")


def _decode(data: bytes, encoding: str | None) -> tuple[str, str]:
    if encoding not in (None, "", "auto"):
        try:
            return data.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError) as error:
            raise ReadError(f"The file cannot be decoded as {encoding}: {error}") from error
    for candidate in _AUTO_ENCODINGS:
        try:
            return data.decode(candidate), candidate
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodes every byte sequence")  # pragma: no cover


def sniff_separator(text: str) -> str:
    """Pick the separator that splits the first lines into the most consistent columns.

    The header decides first: a header rarely contains decimal commas, so a German
    ``Name;Gehalt`` / ``Anna;3,5`` file is read with ``;``, not ``,``.
    """
    lines = [line for line in text.splitlines()[:50] if line.strip()]
    if not lines:
        return ","
    best, best_key = ",", None
    for sep in (",", ";", "\t", "|"):
        widths = [len(row) for row in csv.reader(lines, delimiter=sep)]
        if widths[0] < 2:
            continue
        consistency = sum(width == widths[0] for width in widths) / len(widths)
        key = (consistency, widths[0])
        if best_key is None or key > best_key:
            best, best_key = sep, key
    return best


def _read_text(data: bytes, file_name: str, options: dict) -> tuple[pd.DataFrame, dict]:
    text, encoding = _decode(data, options.get("encoding"))
    sep = _choice(options.get("separator"), SEPARATORS)
    if sep is None:
        sep = "\t" if Path(file_name).suffix.lower() == ".tsv" else sniff_separator(text)
    decimal = _choice(options.get("decimal"), DECIMALS)
    if decimal is None:
        # Semicolon files usually come from a locale that writes 3,5 (German Excel).
        decimal = "," if sep == ";" else "."
    if decimal == sep:
        raise ReadError("The separator and the decimal mark must be different.")
    try:
        frame = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, skipinitialspace=True)
    except pd.errors.EmptyDataError as error:
        raise ReadError("The file has no data.") from error
    except (pd.errors.ParserError, ValueError) as error:
        raise ReadError(f"The file cannot be read as a table: {error}") from error
    return frame, {"separator": sep, "decimal": decimal, "encoding": encoding}


def _read_excel(data: bytes, options: dict) -> tuple[pd.DataFrame, dict]:
    sheet = (options.get("sheet") or "").strip()
    try:
        frame = pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0, engine="openpyxl")
    except ImportError as error:
        raise ReadError("Reading Excel files needs openpyxl: pip install openpyxl") from error
    except (zipfile.BadZipFile, KeyError, OSError) as error:
        raise ReadError("This is not a valid .xlsx file.") from error
    except ValueError as error:
        raise ReadError(f"The Excel file cannot be read: {error}") from error
    return frame, {"sheet": sheet}


def _read_json(data: bytes, *, lines: bool) -> tuple[pd.DataFrame, dict]:
    try:
        frame = pd.read_json(io.BytesIO(data), lines=lines)
    except ValueError as error:
        hint = "one JSON object per line" if lines else 'a list of records, for example [{"a": 1}, {"a": 2}]'
        raise ReadError(f"The JSON file cannot be read as a table; expected {hint}. ({error})") from error
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    return frame, {}


def _read_parquet(data: bytes) -> tuple[pd.DataFrame, dict]:
    try:
        frame = pd.read_parquet(io.BytesIO(data))
    except ImportError as error:
        raise ReadError("Reading Parquet files needs pyarrow: pip install pyarrow") from error
    except (OSError, ValueError) as error:
        raise ReadError(f"The Parquet file cannot be read: {error}") from error
    return frame, {}


def pandas_read_call(file_name: str, read_options: dict) -> str:
    """The pandas call that reads the file the same way, for the generated Python code."""
    kind = read_options.get("format") or file_format(file_name)
    name = repr(file_name)
    if kind == "csv":
        args = [name]
        if read_options.get("separator", ",") != ",":
            args.append(f"sep={read_options['separator']!r}")
        if read_options.get("decimal", ".") != ".":
            args.append(f"decimal={read_options['decimal']!r}")
        if read_options.get("encoding") not in (None, "utf-8", "utf-8-sig"):
            args.append(f"encoding={read_options['encoding']!r}")
        args.append("skipinitialspace=True")
        return f"pd.read_csv({', '.join(args)})"
    if kind == "excel":
        sheet = read_options.get("sheet")
        return f"pd.read_excel({name}, sheet_name={sheet!r})" if sheet else f"pd.read_excel({name})"
    if kind == "jsonl":
        return f"pd.read_json({name}, lines=True)"
    if kind == "json":
        return f"pd.read_json({name})"
    return f"pd.read_parquet({name})"
