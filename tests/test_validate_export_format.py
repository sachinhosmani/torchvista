"""Unit tests for the pure helper validate_export_format."""
import pytest

from torchvista.enums import ExportFormat
from torchvista.render import validate_export_format


class TestValidateExportFormat:
    def test_none_returns_none(self):
        assert validate_export_format(None) is None

    def test_valid_lowercase_returns_enum(self):
        assert validate_export_format("svg") == ExportFormat.SVG
        assert validate_export_format("html") == ExportFormat.HTML
        assert validate_export_format("png") == ExportFormat.PNG

    def test_is_case_insensitive(self):
        assert validate_export_format("HTML") == ExportFormat.HTML
        assert validate_export_format("Svg") == ExportFormat.SVG
        assert validate_export_format("PnG") == ExportFormat.PNG

    def test_invalid_format_raises_value_error(self):
        with pytest.raises(ValueError):
            validate_export_format("pdf")

    def test_error_message_lists_valid_values(self):
        with pytest.raises(ValueError) as excinfo:
            validate_export_format("pdf")
        message = str(excinfo.value)
        assert "pdf" in message
        for valid in ("svg", "html", "png"):
            assert valid in message
