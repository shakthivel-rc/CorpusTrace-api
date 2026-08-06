"""Unit tests for document text extraction.

Before this existed, raw bytes were decoded with errors="ignore" and the FILE STRUCTURE
was indexed — a real PDF produced chunks full of '%PDF-1.4', '/Annots' and 'obj', so every
question scored 0.0 and chat always answered "I do not have enough information...".
These tests assert real text comes out and, just as importantly, that a malformed or
hostile document degrades to a message instead of an unhandled HTTP 500.
"""
import io
import shutil
import zipfile
from pathlib import Path

import pytest

from rag.service import (
    _detect_modality,
    _extract_docx_text,
    _extract_pdf_text,
    _extract_table_text,
    _extract_text,
    _extract_xlsx_text,
    _safe_filename,
)

pytestmark = pytest.mark.unit

DOCX_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _make_docx(paragraphs: list[str]) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    document = f'<?xml version="1.0"?><w:document xmlns:w="{DOCX_NS}"><w:body>{body}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def _make_xlsx(rows: list[list[str]]) -> bytes:
    shared = [cell for row in rows for cell in row]
    shared_xml = (
        f'<?xml version="1.0"?><sst xmlns="{XLSX_NS}">'
        + "".join(f"<si><t>{value}</t></si>" for value in shared)
        + "</sst>"
    )
    index = 0
    row_xml = ""
    for row_number, row in enumerate(rows, start=1):
        cells = ""
        for _ in row:
            cells += f'<c t="s"><v>{index}</v></c>'
            index += 1
        row_xml += f'<row r="{row_number}">{cells}</row>'
    sheet_xml = f'<?xml version="1.0"?><worksheet xmlns="{XLSX_NS}"><sheetData>{row_xml}</sheetData></worksheet>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
    return buffer.getvalue()


def _make_pdf(lines: list[str]) -> bytes:
    """A minimal, valid, uncompressed PDF with a real text layer.

    Built by hand (correct xref offsets) rather than with a writer helper: the page must
    declare a /Font resource with a standard encoding, otherwise a reader cannot map the
    character codes back to text and returns replacement characters.
    """
    content = (
        "BT /F1 12 Tf 72 720 Td "
        + " ".join(f"({line}) Tj 0 -14 Td" for line in lines)
        + " ET"
    ).encode()

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(out)


class TestPdfExtraction:
    def test_returns_the_embedded_text_layer_not_the_file_structure(self):
        pdf = _make_pdf(["Information security management systems", "Requirements"])
        text, failure = _extract_pdf_text(pdf)
        assert failure is None
        assert "Information security management systems" in text
        # The regression: PDF internals must NOT end up in the indexed text.
        assert "%PDF" not in text
        assert "/Annots" not in text

    def test_corrupt_pdf_degrades_to_a_message_instead_of_raising(self):
        # Nothing catches an exception here — an escape becomes an unhandled HTTP 500.
        result = _extract_text("broken.pdf", "application/pdf", b"%PDF-1.4 not really a pdf", "pdf")
        assert "could not be read" in result

    def test_unreadable_pdf_is_not_misreported_as_scanned(self):
        # Telling the user "scanned images / OCR" sends them after a problem they don't have.
        result = _extract_text("broken.pdf", "application/pdf", b"not a pdf at all", "pdf")
        assert "scanned" not in result.lower()
        assert "password-protected" in result

    def test_empty_bytes_do_not_raise(self):
        text, failure = _extract_pdf_text(b"")
        assert text == ""
        assert failure is not None


class TestDocxExtraction:
    def test_extracts_paragraph_text(self):
        docx = _make_docx(["Scope of the standard", "Normative references"])
        text = _extract_docx_text(docx)
        assert "Scope of the standard" in text
        assert "Normative references" in text

    def test_routes_through_extract_text_by_suffix(self):
        docx = _make_docx(["Quality policy"])
        assert "Quality policy" in _extract_text("policy.docx", "application/octet-stream", docx, "text")

    def test_non_zip_payload_degrades_to_a_message(self):
        result = _extract_text("fake.docx", "application/octet-stream", b"this is not a zip", "text")
        assert "No readable text" in result

    def test_zip_without_a_document_part_returns_empty(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("unrelated.txt", "hello")
        assert _extract_docx_text(buffer.getvalue()) == ""

    def test_legacy_doc_reports_that_it_is_unsupported(self):
        result = _extract_text("old.doc", "application/msword", b"\xd0\xcf\x11\xe0binary", "text")
        assert "Legacy .doc parsing is not configured" in result


class TestXlsxExtraction:
    def test_resolves_shared_strings_into_row_text(self):
        xlsx = _make_xlsx([["Control", "Status"], ["A.5.1", "Implemented"]])
        text = _extract_xlsx_text(xlsx)
        assert "Control" in text and "Implemented" in text
        assert text.startswith("Row 1:")

    def test_routes_through_the_table_modality(self):
        xlsx = _make_xlsx([["Risk", "Owner"]])
        assert _detect_modality("register.xlsx", "application/octet-stream") == "table"
        assert "Risk" in _extract_table_text(xlsx, ".xlsx")

    def test_corrupt_workbook_degrades_to_a_message(self):
        assert "no cell text was readable" in _extract_table_text(b"not a zip", ".xlsx")


class TestHostileArchives:
    """Every case here was an unhandled HTTP 500 before hardening. zipfile raises
    RuntimeError / NotImplementedError / zlib.error, none of which share a base class
    with BadZipFile — enumerating them is unmaintainable, so the parsers catch broadly."""

    def _encrypted_docx(self) -> bytes:
        # ZipCrypto: namelist() succeeds, read() raises RuntimeError.
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "word").mkdir()
            (root / "word" / "document.xml").write_text("<a/>")
            archive = root / "evil.docx"
            subprocess.run(
                ["zip", "-q", "-e", "-P", "secret", str(archive), "word/document.xml"],
                cwd=tmp,
                check=True,
            )
            return archive.read_bytes()

    def test_password_protected_docx_does_not_raise(self):
        pytest.importorskip("subprocess")
        if shutil.which("zip") is None:
            pytest.skip("the zip CLI is needed to build a ZipCrypto fixture")
        assert _extract_docx_text(self._encrypted_docx()) == ""

    def test_corrupt_deflate_payload_does_not_raise(self):
        """Header/CRC damage yields BadZipFile, but *payload* damage yields zlib.error —
        the likely real-world case (a truncated upload), and previously a 500."""
        docx = bytearray(_make_docx(["Quality policy statement for the organisation"]))
        for offset in range(60, 100):
            docx[offset] ^= 0xFF
        # Must return a string either way, never propagate.
        assert isinstance(_extract_docx_text(bytes(docx)), str)

    def test_unsupported_compression_method_does_not_raise(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", "<a/>")
        blob = bytearray(buffer.getvalue())
        # Patch the compression method field to an unsupported value (99 = AES).
        blob[8:10] = (99).to_bytes(2, "little")
        assert _extract_docx_text(bytes(blob)) == ""

    def test_declared_zip_bomb_is_refused_without_decompressing(self):
        payload = b"<a>" + b"A" * (40 * 1024 * 1024) + b"</a>"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("word/document.xml", payload)
        assert _extract_docx_text(buffer.getvalue()) == ""

    def test_oversized_csv_field_does_not_raise(self):
        # csv raises _csv.Error above its 128 KB field limit; previously a 500.
        huge = b'name,note\r\nx,"' + b"y" * 200_000 + b'"\r\n'
        assert isinstance(_extract_table_text(huge, ".csv"), str)


class TestExtractionCorrectness:
    def test_docx_tabs_and_breaks_become_separators(self):
        """Without this, runs either side of a tab merge into 'NameJohn' — a token no
        query can match, so the content is effectively unsearchable."""
        document = (
            f'<?xml version="1.0"?><w:document xmlns:w="{DOCX_NS}"><w:body>'
            "<w:p><w:r><w:t>Name</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>John</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", document)
        assert "NameJohn" not in _extract_docx_text(buffer.getvalue())

    def test_docx_text_box_paragraph_is_not_duplicated(self):
        inner = "<w:p><w:r><w:t>Callout text</w:t></w:r></w:p>"
        document = (
            f'<?xml version="1.0"?><w:document xmlns:w="{DOCX_NS}"><w:body>'
            f"<w:p><w:r><w:txbxContent>{inner}</w:txbxContent></w:r></w:p>"
            "</w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", document)
        assert _extract_docx_text(buffer.getvalue()).count("Callout text") == 1

    def test_xlsx_sheets_are_ordered_numerically_not_lexicographically(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for number in (1, 2, 10):
                sheet = (
                    f'<?xml version="1.0"?><worksheet xmlns="{XLSX_NS}"><sheetData>'
                    f'<row r="1"><c t="inlineStr"><is><t>SHEET{number}</t></is></c></row>'
                    "</sheetData></worksheet>"
                )
                archive.writestr(f"xl/worksheets/sheet{number}.xml", sheet)
        text = _extract_xlsx_text(buffer.getvalue())
        assert text.index("SHEET2") < text.index("SHEET10")

    def test_xlsx_uses_the_real_row_index(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            sheet = (
                f'<?xml version="1.0"?><worksheet xmlns="{XLSX_NS}"><sheetData>'
                '<row r="7"><c t="inlineStr"><is><t>Late row</t></is></c></row>'
                "</sheetData></worksheet>"
            )
            archive.writestr("xl/worksheets/sheet1.xml", sheet)
        assert "Row 7:" in _extract_xlsx_text(buffer.getvalue())

    def test_xlsx_booleans_are_readable_words(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            sheet = (
                f'<?xml version="1.0"?><worksheet xmlns="{XLSX_NS}"><sheetData>'
                '<row r="1"><c t="b"><v>1</v></c><c t="b"><v>0</v></c></row>'
                "</sheetData></worksheet>"
            )
            archive.writestr("xl/worksheets/sheet1.xml", sheet)
        text = _extract_xlsx_text(buffer.getvalue())
        assert "TRUE" in text and "FALSE" in text

    def test_one_bad_shared_string_index_does_not_discard_the_workbook(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "xl/sharedStrings.xml",
                f'<?xml version="1.0"?><sst xmlns="{XLSX_NS}"><si><t>Good value</t></si></sst>',
            )
            sheet = (
                f'<?xml version="1.0"?><worksheet xmlns="{XLSX_NS}"><sheetData>'
                '<row r="1"><c t="s"><v>not-a-number</v></c><c t="s"><v>0</v></c></row>'
                "</sheetData></worksheet>"
            )
            archive.writestr("xl/worksheets/sheet1.xml", sheet)
        assert "Good value" in _extract_xlsx_text(buffer.getvalue())

    def test_strict_ooxml_namespace_still_parses(self):
        strict_ns = "http://purl.oclc.org/ooxml/wordprocessingml/main"
        document = (
            f'<?xml version="1.0"?><w:document xmlns:w="{strict_ns}"><w:body>'
            "<w:p><w:r><w:t>Strict namespace body</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", document)
        assert "Strict namespace body" in _extract_docx_text(buffer.getvalue())

    def test_legacy_xls_is_not_indexed_as_binary_garbage(self):
        # Decoding OLE2 bytes indexed junk as content — the same class of bug as the PDF one.
        result = _extract_table_text(b"\xd0\xcf\x11\xe0 Workbook binary payload data", ".xls")
        assert "Legacy .xls parsing is not configured" in result


class TestSafeFilenameExtensions:
    def test_non_ascii_stem_keeps_its_extension(self):
        """If the extension is lost the suffix dispatch never runs and the document is
        indexed as raw bytes — the exact failure this whole change removes."""
        assert _safe_filename("文档.pdf").endswith(".pdf")
        assert _safe_filename("отчет.docx").endswith(".docx")

    def test_extensionless_names_still_work(self):
        assert _safe_filename("README") == "README"


class TestModalityRouting:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("a.pdf", "pdf"),
            ("a.docx", "text"),
            ("a.xlsx", "table"),
            ("a.csv", "table"),
            ("a.png", "image"),
            ("a.mp3", "audio"),
            ("a.txt", "text"),
        ],
    )
    def test_detects_the_expected_modality(self, filename, expected):
        assert _detect_modality(filename, "application/octet-stream") == expected
