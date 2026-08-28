"""Tests for SNP list I/O functions.

Validates read_snp_list_file and resolve_snp_list_to_indices for
edge cases including whitespace handling, empty files, partial matches,
and case sensitivity.
"""

import numpy as np
import pytest

from jamma.io.snp_list import read_snp_list_file, resolve_snp_list_to_indices

pytestmark = pytest.mark.tier0


class TestReadSnpListFile:
    """Tests for read_snp_list_file."""

    @pytest.mark.parametrize(
        "content,expected",
        [
            pytest.param(
                "rs001\nrs002\nrs003\n",
                {"rs001", "rs002", "rs003"},
                id="basic",
            ),
            pytest.param(
                "  rs001  \n\trs002\t\n rs003 \n",
                {"rs001", "rs002", "rs003"},
                id="strips_whitespace",
            ),
            pytest.param(
                "rs001\n\n\nrs002\n\nrs003\n",
                {"rs001", "rs002", "rs003"},
                id="skips_empty_lines",
            ),
            pytest.param(
                "rs123 1 A T\nrs456 2 G C\nrs789 3 T A\n",
                {"rs123", "rs456", "rs789"},
                id="first_token_only",
            ),
        ],
    )
    def test_read_snp_list_parsing(self, tmp_path, content, expected):
        """read_snp_list_file handles whitespace, empty lines, and extra columns."""
        snp_file = tmp_path / "snps.txt"
        snp_file.write_text(content)

        result = read_snp_list_file(snp_file)
        assert result == expected

    def test_read_snp_list_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            read_snp_list_file(tmp_path / "nonexistent.txt")

    def test_read_snp_list_empty_file(self, tmp_path):
        """Raises ValueError for empty file."""
        snp_file = tmp_path / "snps.txt"
        snp_file.write_text("")

        with pytest.raises(ValueError, match="empty or contains no valid IDs"):
            read_snp_list_file(snp_file)

    def test_read_snp_list_whitespace_only_file(self, tmp_path):
        """Raises ValueError for file containing only whitespace."""
        snp_file = tmp_path / "snps.txt"
        snp_file.write_text("  \n\n  \n")

        with pytest.raises(ValueError, match="empty or contains no valid IDs"):
            read_snp_list_file(snp_file)


class TestResolveSnpListToIndices:
    """Tests for resolve_snp_list_to_indices."""

    def test_resolve_snp_list_to_indices(self):
        """5 BIM sids, request 3, get sorted indices."""
        bim_sids = np.array(["rs001", "rs002", "rs003", "rs004", "rs005"])
        snp_ids = {"rs002", "rs004", "rs005"}

        result = resolve_snp_list_to_indices(snp_ids, bim_sids)
        np.testing.assert_array_equal(result, [1, 3, 4])
        assert result.dtype == np.intp

    def test_resolve_snp_list_partial_match(self):
        """Request 5, only 3 found, returns matched indices."""
        bim_sids = np.array(["rs001", "rs002", "rs003", "rs004", "rs005"])
        snp_ids = {"rs001", "rs003", "rs005", "rs999", "rs888"}

        result = resolve_snp_list_to_indices(snp_ids, bim_sids)
        np.testing.assert_array_equal(result, [0, 2, 4])
        # Warning is logged via loguru (verified in captured stderr output)

    def test_resolve_snp_list_zero_match(self):
        """None found raises ValueError."""
        bim_sids = np.array(["rs001", "rs002", "rs003"])
        snp_ids = {"rs999", "rs888"}

        with pytest.raises(ValueError, match="No SNPs from the list matched"):
            resolve_snp_list_to_indices(snp_ids, bim_sids)

    def test_resolve_snp_list_case_sensitive(self):
        """RS001 vs rs001 -- no match (case sensitive)."""
        bim_sids = np.array(["rs001", "rs002", "rs003"])
        snp_ids = {"RS001", "RS002"}

        with pytest.raises(ValueError, match="No SNPs from the list matched"):
            resolve_snp_list_to_indices(snp_ids, bim_sids)
