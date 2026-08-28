"""Cache-key derivation and manifest validation for the eigen cache.

``compute_eigen_cache_key`` must change exactly when a recomputed eigen pair
would differ, and the manifest is what turns that key into a stale-cache
decision at read time. Both work on files this module writes itself, so
nothing here touches the mouse_hs1940 fixture the LOCO tests read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.lmm.eigen_cache import EigenCacheComponents

pytestmark = pytest.mark.tier0


def _dummy_components(maf_threshold: float = 0.01) -> EigenCacheComponents:
    """Build a complete EigenCacheComponents for manifest write tests.

    The manifest tests care about the cache_key, not the components payload,
    so a valid fully-populated default keeps the call sites type-clean.
    maf_threshold is exposed because the roundtrip test asserts on it.
    """
    return {
        "schema_version": 1,
        "bed_fingerprint": "data.bed:64:1",
        "bim_sha256": "0" * 64,
        "maf_threshold": maf_threshold,
        "miss_threshold": 0.05,
        "valid_mask_sha256": "1" * 64,
        "ksnps": "none",
    }


def _write_dummy_plink(
    prefix: Path,
    *,
    bed_size: int = 64,
    bim_lines: list[str] | None = None,
    bed_fill: int = 0,
) -> None:
    """Write minimal .bed/.bim files at ``prefix`` for cache-key unit tests.

    The cache-key function only stats .bed (name + size + mtime) and hashes
    .bim content, so these need not be valid PLINK binaries.
    """
    if bim_lines is None:
        bim_lines = [
            "1\trs1\t0\t100\tA\tG",
            "1\trs2\t0\t200\tC\tT",
            "2\trs3\t0\t300\tA\tT",
        ]
    prefix.with_suffix(".bed").write_bytes(bytes([bed_fill]) * bed_size)
    prefix.with_suffix(".bim").write_text("\n".join(bim_lines) + "\n")


def _compute_key(
    prefix: Path,
    *,
    maf_threshold: float = 0.01,
    miss_threshold: float = 0.05,
    valid_mask: np.ndarray | None = None,
    ksnps_indices: np.ndarray | None = None,
) -> str:
    """Call compute_eigen_cache_key with per-test overrides over fixed defaults.

    Explicit keyword forwarding (not dict-unpacking) keeps the call type-clean.
    """
    from jamma.lmm.eigen_cache import compute_eigen_cache_key

    if valid_mask is None:
        valid_mask = np.ones(20, dtype=bool)
    key, _components = compute_eigen_cache_key(
        prefix,
        maf_threshold=maf_threshold,
        miss_threshold=miss_threshold,
        valid_mask=valid_mask,
        ksnps_indices=ksnps_indices,
    )
    return key


class TestEigenCacheKey:
    """compute_eigen_cache_key changes iff a real eigen-pair determinant changes."""

    def test_key_is_stable_for_identical_inputs(self, tmp_path: Path) -> None:
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        k1 = _compute_key(prefix)
        k2 = _compute_key(prefix)
        assert isinstance(k1, str)
        assert len(k1) > 0
        assert k1 == k2

    def test_key_changes_when_maf_threshold_changes(self, tmp_path: Path) -> None:
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        assert _compute_key(prefix) != _compute_key(prefix, maf_threshold=0.05)

    def test_key_changes_when_miss_threshold_changes(self, tmp_path: Path) -> None:
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        assert _compute_key(prefix) != _compute_key(prefix, miss_threshold=0.10)

    def test_key_changes_when_valid_mask_positions_change(self, tmp_path: Path) -> None:
        """Same valid COUNT, different valid POSITIONS -> different key.

        This is the silent-stale hole the manifest closes: two phenotypes with
        the same number of non-missing samples but a different missingness
        pattern select a different sample subset, hence a different K.
        """
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        m1 = np.ones(20, dtype=bool)
        m1[0] = False
        m2 = np.ones(20, dtype=bool)
        m2[1] = False
        assert int(m1.sum()) == int(m2.sum())  # same count
        k1 = _compute_key(prefix, valid_mask=m1)
        k2 = _compute_key(prefix, valid_mask=m2)
        assert k1 != k2

    def test_key_changes_when_valid_mask_length_changes(self, tmp_path: Path) -> None:
        """Different total sample count (.fam size) -> different key."""
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        k1 = _compute_key(prefix, valid_mask=np.ones(20, dtype=bool))
        k2 = _compute_key(prefix, valid_mask=np.ones(19, dtype=bool))
        assert k1 != k2

    def test_key_changes_when_bim_content_changes(self, tmp_path: Path) -> None:
        """Re-annotating a SNP's chromosome changes the LOCO partition -> key."""
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        k1 = _compute_key(prefix)
        _write_dummy_plink(
            prefix,
            bim_lines=[
                "1\trs1\t0\t100\tA\tG",
                "1\trs2\t0\t200\tC\tT",
                "3\trs3\t0\t300\tA\tT",  # chr 2 -> 3
            ],
        )
        assert k1 != _compute_key(prefix)

    def test_key_changes_when_bed_content_changes(self, tmp_path: Path) -> None:
        """A different .bed (here: different size) -> different key."""
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix, bed_size=64)
        k1 = _compute_key(prefix)
        _write_dummy_plink(prefix, bed_size=128)
        assert k1 != _compute_key(prefix)

    def test_key_changes_when_ksnps_changes(self, tmp_path: Path) -> None:
        """Different kinship-SNP restriction -> different key; None differs too."""
        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        k_none = _compute_key(prefix)
        k_a = _compute_key(prefix, ksnps_indices=np.array([0, 1]))
        k_b = _compute_key(prefix, ksnps_indices=np.array([0, 2]))
        assert k_none != k_a
        assert k_none != k_b
        assert k_a != k_b

    def test_returns_canonical_components(self, tmp_path: Path) -> None:
        """Second return value is the exact hashed payload (for the manifest)."""
        import hashlib
        import json

        from jamma.lmm.eigen_cache import compute_eigen_cache_key

        prefix = tmp_path / "data"
        _write_dummy_plink(prefix)
        key, components = compute_eigen_cache_key(
            prefix,
            maf_threshold=0.01,
            miss_threshold=0.05,
            valid_mask=np.ones(20, dtype=bool),
        )
        assert isinstance(components, dict)
        hashed_keys = {"bed_fingerprint", "bim_sha256", "valid_mask_sha256"}
        assert hashed_keys <= components.keys()
        canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert key == expected


class TestEigenCacheManifest:
    """Manifest read/write/validate behavior for stale-cache detection."""

    def test_absent_manifest_is_invalid(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import eigen_cache_is_valid

        ok, reason = eigen_cache_is_valid(tmp_path, "result", "somekey")
        assert ok is False
        assert "manifest" in reason.lower()

    def test_matching_key_is_valid(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import (
            eigen_cache_is_valid,
            write_eigen_cache_manifest,
        )

        write_eigen_cache_manifest(
            tmp_path, "result", "KEY123", components=_dummy_components()
        )
        ok, _reason = eigen_cache_is_valid(tmp_path, "result", "KEY123")
        assert ok is True

    def test_mismatched_key_is_invalid(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import (
            eigen_cache_is_valid,
            write_eigen_cache_manifest,
        )

        write_eigen_cache_manifest(
            tmp_path, "result", "KEY123", components=_dummy_components()
        )
        ok, reason = eigen_cache_is_valid(tmp_path, "result", "DIFFERENT")
        assert ok is False
        assert reason

    def test_manifest_roundtrip(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import (
            read_eigen_cache_manifest,
            write_eigen_cache_manifest,
        )

        components = _dummy_components(maf_threshold=0.01)
        path = write_eigen_cache_manifest(
            tmp_path, "result", "KEY123", components=components
        )
        assert path.exists()
        manifest = read_eigen_cache_manifest(tmp_path, "result")
        assert manifest is not None
        assert manifest["cache_key"] == "KEY123"
        assert manifest["components"] == components

    def test_corrupt_manifest_is_invalid(self, tmp_path: Path) -> None:
        from jamma.lmm.eigen_cache import (
            eigen_cache_is_valid,
            eigen_cache_manifest_path,
        )

        eigen_cache_manifest_path(tmp_path, "result").write_text("{ not json")
        ok, reason = eigen_cache_is_valid(tmp_path, "result", "KEY")
        assert ok is False
        assert reason

    def test_missing_cache_key_reports_malformed_not_input_change(
        self, tmp_path: Path
    ) -> None:
        """A manifest that parses but lacks cache_key is malformed, not stale.

        Mislabeling it 'inputs changed' misdirects anyone debugging an
        unexpected invalidation, so the reason must name the malformed manifest.
        """
        import json

        from jamma.lmm.eigen_cache import (
            EIGEN_CACHE_SCHEMA_VERSION,
            eigen_cache_is_valid,
            eigen_cache_manifest_path,
        )

        path = eigen_cache_manifest_path(tmp_path, "result")
        # Valid schema_version so the schema gate passes and we reach the
        # missing-cache_key branch; no cache_key field.
        path.write_text(
            json.dumps(
                {
                    "schema_version": EIGEN_CACHE_SCHEMA_VERSION,
                    "components": {},
                }
            )
        )
        ok, reason = eigen_cache_is_valid(tmp_path, "result", "KEY")
        assert ok is False
        assert "malformed" in reason.lower()
        assert "cache_key" in reason
        assert "inputs changed" not in reason

    def test_schema_version_mismatch_reports_schema_reason(
        self, tmp_path: Path
    ) -> None:
        """A manifest with a stale schema_version is rejected explicitly.

        The cache_key matches, so only the explicit schema gate (not the
        implicit hash coupling) can produce this rejection.
        """
        import json

        from jamma.lmm.eigen_cache import (
            EIGEN_CACHE_SCHEMA_VERSION,
            eigen_cache_is_valid,
            eigen_cache_manifest_path,
        )

        path = eigen_cache_manifest_path(tmp_path, "result")
        path.write_text(
            json.dumps(
                {
                    "schema_version": EIGEN_CACHE_SCHEMA_VERSION + 1,
                    "cache_key": "KEY",
                    "components": {},
                }
            )
        )
        ok, reason = eigen_cache_is_valid(tmp_path, "result", "KEY")
        assert ok is False
        assert "schema_version" in reason

    def test_write_failure_leaves_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A serialisation failure mid-write must leave no half-written artifact.

        json.dump is an I/O boundary, safe to patch; the guarantee under test is
        that the temp file is cleaned up and no manifest is left behind.
        """
        import json as json_mod

        from jamma.lmm.eigen_cache import (
            eigen_cache_manifest_path,
            write_eigen_cache_manifest,
        )

        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated serialisation failure")

        monkeypatch.setattr(json_mod, "dump", boom)

        with pytest.raises(RuntimeError, match="simulated serialisation failure"):
            write_eigen_cache_manifest(
                tmp_path, "result", "KEY", components=_dummy_components()
            )

        assert list(tmp_path.glob("*.json")) == []
        assert not eigen_cache_manifest_path(tmp_path, "result").exists()

    def test_invalidate_removes_present_manifest_and_no_ops_when_absent(
        self, tmp_path: Path
    ) -> None:
        from jamma.lmm.eigen_cache import (
            eigen_cache_manifest_path,
            invalidate_eigen_cache_manifest,
            write_eigen_cache_manifest,
        )

        write_eigen_cache_manifest(
            tmp_path, "result", "KEY", components=_dummy_components()
        )
        manifest = eigen_cache_manifest_path(tmp_path, "result")
        assert manifest.exists()

        invalidate_eigen_cache_manifest(tmp_path, "result")
        assert manifest.exists() is False

        invalidate_eigen_cache_manifest(tmp_path, "result")
        assert manifest.exists() is False
