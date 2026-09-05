"""Early row selection must preserve full-population kinship preprocessing."""

from functools import partial
from pathlib import Path

import numpy as np
import pytest
from bed_reader import to_bed
from loguru import logger

from jamma.kinship import compute_kinship_streaming, compute_loco_kinship_streaming
from jamma.pipeline import PipelineConfig, PipelineRunner
from jamma.validation.compare import load_gemma_assoc
from tests.conftest import require_fixture
from tests.fixture_paths import LOCO, SYNTHETIC


@pytest.fixture
def asymmetric_plink(tmp_path: Path) -> Path:
    """SNPs cross MAF/missingness/monomorphism thresholds when rows are dropped."""
    rng = np.random.default_rng(327)
    genotypes = rng.binomial(2, np.linspace(0.05, 0.5, 61), (80, 61)).astype(float)
    genotypes[rng.random(genotypes.shape) < 0.04] = np.nan
    genotypes[:40, 0] = 0  # Polymorphic only outside the retained population.
    genotypes[40:, 0] = 2
    genotypes[:40, 1] = np.nan  # Missingness differs between populations.
    genotypes[:, 2] = 1  # Globally monomorphic.
    genotypes[:, 3] = np.nan  # Globally missing.
    bfile = tmp_path / "asymmetric"
    to_bed(
        bfile.with_suffix(".bed"),
        genotypes,
        properties={"chromosome": ["1"] * 20 + ["2"] * 20 + ["3"] * 21},
    )
    return bfile


@pytest.mark.tier0
@pytest.mark.parametrize("mode", ["centered", "standardized"])
@pytest.mark.parametrize("chunk_size", [7, 100])
@pytest.mark.parametrize("filter_subset", [False, True])
@pytest.mark.parametrize(
    "maf,miss,restrict", [(0.0, 1.0, False), (0.3, 0.1, False), (0.0, 1.0, True)]
)
def test_early_rows_equal_full_kinship_slice(
    asymmetric_plink, mode, chunk_size, maf, miss, restrict, filter_subset
):
    valid = np.arange(40)
    compute = partial(
        compute_kinship_streaming,
        asymmetric_plink,
        mode=mode,
        filter_sample_indices=valid if filter_subset else None,
        chunk_size=chunk_size,
        maf_threshold=maf,
        miss_threshold=miss,
        ksnps_indices=np.arange(0, 61, 2) if restrict else None,
        check_memory=False,
        show_progress=False,
    )
    full = compute()
    subset = compute(valid_indices=valid)
    assert subset.shape == (40, 40)
    np.testing.assert_allclose(
        subset, full[np.ix_(valid, valid)], rtol=1e-12, atol=1e-14
    )


@pytest.mark.tier0
@pytest.mark.parametrize("batch_chrs", [1, 3])
@pytest.mark.parametrize("filter_subset", [False, True])
@pytest.mark.parametrize(
    "maf,miss,restrict", [(0.0, 1.0, False), (0.3, 0.1, False), (0.0, 1.0, True)]
)
def test_early_rows_equal_full_loco_slices(
    asymmetric_plink, batch_chrs, maf, miss, restrict, filter_subset
):
    valid = np.arange(40)
    compute = partial(
        compute_loco_kinship_streaming,
        asymmetric_plink,
        chunk_size=7,
        maf_threshold=maf,
        miss_threshold=miss,
        ksnps_indices=np.arange(0, 61, 2) if restrict else None,
        check_memory=False,
        show_progress=False,
        _max_batch_chrs=batch_chrs,
        filter_sample_indices=valid if filter_subset else None,
    )
    full = compute().materialize()
    subset = compute(valid_indices=valid).materialize()
    assert full.keys() == subset.keys()
    for chromosome in full:
        np.testing.assert_allclose(
            subset[chromosome],
            full[chromosome][np.ix_(valid, valid)],
            rtol=1e-12,
            atol=1e-14,
        )


def _assert_same_associations(left, right):
    assert len(left) > 0
    assert [r.rs for r in left] == [r.rs for r in right]
    for field in [
        "af",
        "n_miss",
        "beta",
        "se",
        "logl_H1",
        "l_remle",
        "l_mle",
        "p_wald",
        "p_lrt",
        "p_score",
    ]:
        np.testing.assert_allclose(
            [getattr(r, field) for r in left],
            [getattr(r, field) for r in right],
            rtol=1e-7,
            atol=1e-12,
            err_msg=field,
        )


@pytest.mark.tier1
@pytest.mark.parametrize("loco", [False, True])
@pytest.mark.parametrize("missing", ["phenotype", "covariate"])
@pytest.mark.parametrize("backend", ["numpy", "numpy-streaming"])
def test_saving_kinship_preserves_associations(tmp_path, loco, missing, backend):
    import shutil

    source = LOCO if loco else SYNTHETIC
    require_fixture(source.bed, source.bim, source.fam)
    bfile = tmp_path / "masked"
    for suffix in [".bed", ".bim", ".fam"]:
        shutil.copyfile(source.bfile.with_suffix(suffix), bfile.with_suffix(suffix))
    covariate_file = None
    if missing == "phenotype":
        fam = np.loadtxt(bfile.with_suffix(".fam"), dtype=str)
        fam[50:, 5] = "-9"
        np.savetxt(bfile.with_suffix(".fam"), fam, fmt="%s")
    else:
        covariate_file = tmp_path / "covariates.txt"
        covariates = np.column_stack(
            [np.ones(100), np.random.default_rng(4).normal(size=100)]
        )
        covariates[50:, 1] = np.nan
        np.savetxt(covariate_file, covariates)

    results = []
    for save in [False, True]:
        run = PipelineRunner(
            PipelineConfig(
                bfile=bfile,
                output_dir=tmp_path,
                output_prefix=f"save_{save}",
                save_kinship=save,
                backend=backend,
                loco=loco,
                covariate_file=covariate_file,
                lmm_mode=4,
                maf=0.3,
                check_memory=False,
                show_progress=False,
                no_telemetry=True,
            )
        ).run()
        results.append(load_gemma_assoc(run.assoc_path))
    _assert_same_associations(*results)


@pytest.mark.tier1
@pytest.mark.parametrize("save_first", [False, True])
@pytest.mark.parametrize("stale_cache", [False, True])
def test_loco_cache_reuse_preserves_save_flag_parity(tmp_path, save_first, stale_cache):
    import json
    import shutil

    from jamma.lmm.eigen_cache import eigen_cache_manifest_path

    require_fixture(LOCO.bed, LOCO.bim, LOCO.fam)
    bfile = tmp_path / "masked"
    for suffix in [".bed", ".bim", ".fam"]:
        shutil.copyfile(LOCO.bfile.with_suffix(suffix), bfile.with_suffix(suffix))
    fam = np.loadtxt(bfile.with_suffix(".fam"), dtype=str)
    fam[50:, 5] = "-9"
    np.savetxt(bfile.with_suffix(".fam"), fam, fmt="%s")
    results = []
    for name, save, write, eigen_dir in [
        ("write", save_first, True, tmp_path / "cache"),
        ("read", not save_first, False, tmp_path / "cache"),
        ("fresh", not save_first, False, None),
    ]:
        if name == "read" and stale_cache:
            manifest_path = eigen_cache_manifest_path(tmp_path / "cache", "study")
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = 1
            manifest_path.write_text(json.dumps(manifest))
        messages = []
        sink = logger.add(
            lambda message, messages=messages: messages.append(str(message))
        )
        try:
            run = PipelineRunner(
                PipelineConfig(
                    bfile=bfile,
                    output_dir=tmp_path,
                    output_prefix="study",
                    loco=True,
                    save_kinship=save,
                    write_eigen=write,
                    eigen_dir=eigen_dir,
                    lmm_mode=4,
                    maf=0.3,
                    check_memory=False,
                    show_progress=False,
                    no_telemetry=True,
                )
            ).run()
        finally:
            logger.remove(sink)
        if name == "read":
            expected = (
                "schema_version 1" if stale_cache else "Found complete LOCO eigen cache"
            )
            assert any(expected in message for message in messages)
        results.append(load_gemma_assoc(run.assoc_path))
    _assert_same_associations(results[0], results[1])
    _assert_same_associations(results[0], results[2])


@pytest.mark.tier1
@pytest.mark.parametrize("save", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_pipeline_kinship_uses_gemma_filter_and_centring_populations(
    asymmetric_plink, tmp_path, save, weighted
):
    """GEMMA filters on analysed samples, then centres over the full population."""
    import warnings

    from jamma.io import load_plink_binary
    from jamma.pipeline_plan import ComputedKinship

    genotypes = load_plink_binary(asymmetric_plink).genotypes.astype(np.float64)
    valid = np.arange(40)
    analysed = genotypes[valid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        af = np.nanmean(analysed, axis=0) / 2
        variance = np.nanvar(analysed, axis=0)
    selected = (
        (np.minimum(af, 1 - af) >= 0.3)
        & (np.isnan(analysed).mean(axis=0) <= 0.1)
        & (variance > 0)
    )
    assert selected.any()
    columns = genotypes[:, selected]
    means = np.nanmean(columns, axis=0)
    centred = np.where(np.isnan(columns), means, columns) - means
    analysed_centred = centred[valid] - centred[valid].mean(axis=0)
    expected = analysed_centred @ analysed_centred.T / selected.sum()
    weight_file = None
    selected_weights = None
    if weighted:
        weights = np.linspace(0.5, 2.0, 80)
        weight_file = tmp_path / "weights.txt"
        np.savetxt(weight_file, weights)
        selected_weights = weights[valid]
        expected /= np.sqrt(weights[valid, None] * weights[None, valid])
    runner = PipelineRunner(
        PipelineConfig(
            bfile=asymmetric_plink,
            weight_file=weight_file,
            output_dir=tmp_path,
            save_kinship=save,
            maf=0.3,
            miss=0.1,
            check_memory=False,
            show_progress=False,
            no_telemetry=True,
        )
    )
    actual = runner._load_kinship_from_source(
        ComputedKinship(None), 80, valid, selected_weights
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)

    if save:
        saved = np.load(tmp_path / "result.cXX.npy")
        np.testing.assert_allclose(
            saved, centred @ centred.T / selected.sum(), rtol=1e-12, atol=1e-14
        )
