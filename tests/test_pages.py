import pytest

from pdfcloak import PdfCloakError
from pdfcloak.pages import parse_page_spec


@pytest.mark.parametrize("spec,expected", [
    ("all", [0, 1, 2, 3, 4]),
    (None, [0, 1, 2, 3, 4]),
    ("3", [2]),
    ("3-5", [2, 3, 4]),
    ("3-", [2, 3, 4]),
    ("-2", [0, 1]),
    ("1,4,last", [0, 3, 4]),
    ("first", [0]),
])
def test_parse_ok(spec, expected):
    assert parse_page_spec(spec, 5) == expected


@pytest.mark.parametrize("spec", ["5-3", "6", "0", "x", "1,,2", ""])
def test_parse_bad(spec):
    if spec == "":
        assert parse_page_spec(spec, 5) == [0, 1, 2, 3, 4]
    else:
        with pytest.raises(PdfCloakError):
            parse_page_spec(spec, 5)
