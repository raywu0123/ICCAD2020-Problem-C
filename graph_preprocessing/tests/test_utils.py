import pytest

from ..utils import extract_bitwidth
from ..constants import UNDEF_USAGE_BITWIDTH


@pytest.mark.parametrize(
    "test_case, ans", [
        ("w[0]", ("w", (0, 0))),
        ("w[10:0]", ("w", (10, 0))),
        ("w[0:10]", ("w", (0, 10))),
        ("w", ("w", UNDEF_USAGE_BITWIDTH)),
        ("1'b1", ("1'b1", UNDEF_USAGE_BITWIDTH)),
    ]
)
def test_extract_bitwidth(test_case, ans):
    name, bitwidth = extract_bitwidth(test_case)

    assert name == ans[0]
    assert bitwidth == ans[1]