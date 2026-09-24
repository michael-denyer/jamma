"""BGEN input and the ``-p`` phenotype file through the CLI and ``gwas()``.

Expected SNP sets and kinship matrices come from the reference decoder
(``tests/reference/bgen.py``), the GCTA INFO oracle (``tests/reference/info.py``)
and the reference kinship (``tests/reference/kinship.py``), so no production
filter sits on the expected side.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from loguru import logger

from jamma.cli import main
from jamma.gwas import gwas
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.eigen_cache import eigen_cache_manifest_path, resolve_eigen_cache
from jamma.pipeline import PipelineConfig, PipelineRunner
from jamma.validation.compare import load_gemma_assoc
from tests.bgen_files import BgenFiles, random_probabilities, write_bgen
from tests.fixture_paths import SYNTHETIC
from tests.reference.bgen import decode_file
from tests.reference.info import gcta_info
from tests.reference.kinship import compute_centered_kinship
from tests.support import require_fixture, requires_c

pytestmark = [pytest.mark.tier0, requires_c]

runner = CliRunner()

N_SAMPLES = 150
PER_CHR = 20
CHROMOSOMES = ["1"] * PER_CHR + ["2"] * PER_CHR + ["3"] * PER_CHR
N_VARIANTS = len(CHROMOSOMES)
MAF = 0.01
MISS = 0.05
INFO = 0.8


@pytest.fixture
def bgen_files(tmp_path: Path) -> BgenFiles:
    """Probabilities blending hard calls with noise; INFO spans each chromosome."""
    rng = np.random.default_rng(7)
    noise = random_probabilities(rng, N_VARIANTS, N_SAMPLES, missing_rate=0.01)
    calls = rng.binomial(2, rng.uniform(0.15, 0.5, (N_VARIANTS, 1)), noise.shape[:2])
    certainty = np.tile(np.linspace(0.2, 1.0, PER_CHR), 3)[:, None, None]
    probs = certainty * np.eye(3)[calls] + (1 - certainty) * noise
    (tmp_path / "data").mkdir()
    return write_bgen(
        tmp_path / "data" / "imputed.bgen",
        probs,
        chromosomes=CHROMOSOMES,
        alleles=[("C", "T")] * N_VARIANTS,
    )


def _write_phenotypes(path: Path, missing: tuple[int, ...]) -> Path:
    """Two columns; column 1 is ``NA`` or ``-9`` on the ``missing`` rows."""
    rng = np.random.default_rng(11)
    lines = []
    for i in range(N_SAMPLES):
        first = ("NA", "-9")[i % 2] if i in missing else f"{rng.normal():.6f}"
        lines.append(f"{first}\t{rng.normal():.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def pheno(bgen_files: BgenFiles) -> Path:
    return _write_phenotypes(bgen_files.bgen.parent / "pheno.txt", (0, 3, 8, 9))


def _analysed_rows() -> np.ndarray:
    return np.setdiff1d(np.arange(N_SAMPLES), [0, 3, 8, 9])


def _reference_keep(
    files: BgenFiles, rows: np.ndarray, *, info_threshold: float
) -> np.ndarray:
    """MAF, missingness, polymorphism and INFO, measured over ``rows``."""
    decoded = decode_file(files.bgen)
    x = decoded.dosages[rows, :]
    miss = np.isnan(x).mean(axis=0)
    af = np.nanmean(x, axis=0) / 2
    maf = np.minimum(af, 1 - af)
    info = np.array(
        [
            gcta_info(
                decoded.q11[:, j],
                decoded.q12[:, j],
                decoded.missing[:, j],
                int(decoded.bit_depth[j]),
                rows,
            )
            for j in range(N_VARIANTS)
        ]
    )
    keep = (maf >= MAF) & (miss <= MISS) & (np.nanvar(x, axis=0) > 0)
    return keep & (info >= info_threshold) if info_threshold > 0 else keep


def _invoke(args: list[str]) -> str:
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    return result.output


def _bgen_args(files: BgenFiles, pheno: Path) -> list[str]:
    return ["-bgen", str(files.bgen), "-p", str(pheno)]


def _tested(path: Path) -> list[str]:
    return [r.rs for r in load_gemma_assoc(path)]


# --------------------------------------------------------------------------
# End to end on BGEN
# --------------------------------------------------------------------------


@pytest.mark.parametrize("info", [0.0, INFO])
def test_gk_then_lmm_tests_the_reference_snp_set(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path, info: float
):
    """``-gk 1`` then ``-lmm 1 -k`` on BGEN, with and without ``-info``.

    The kinship spans every sample over the SNPs the filters keep on the
    analysed rows, and the association tests exactly the reference set.
    """
    out = tmp_path / "out"
    info_args = ["-info", str(info)] if info > 0 else []
    base = [*_bgen_args(bgen_files, pheno), *info_args, "-outdir", str(out)]
    _invoke(["-gk", "1", *base, "-o", "kin"])
    _invoke(["-lmm", "1", *base, "-k", str(out / "kin.cXX.npy"), "-o", "assoc"])

    rows = _analysed_rows()
    keep = _reference_keep(bgen_files, rows, info_threshold=info)
    if info > 0:
        assert (
            0 < keep.sum() < _reference_keep(bgen_files, rows, info_threshold=0).sum()
        )
    rs = np.array([f"rs{j}" for j in range(N_VARIANTS)])
    assert _tested(out / "assoc.assoc.txt") == list(rs[keep])

    dosages = decode_file(bgen_files.bgen).dosages
    expected_k = compute_centered_kinship(dosages[:, keep], check_memory=False)
    np.testing.assert_allclose(
        read_kinship_matrix(out / "kin.cXX.npy", n_samples=N_SAMPLES),
        expected_k,
        rtol=1e-10,
        atol=1e-14,
    )


def test_gwas_computed_kinship_uses_the_info_filter(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path
):
    """``gwas()`` without ``-k`` computes kinship over the INFO-filtered SNPs."""
    result = gwas(
        bgen=bgen_files.bgen,
        phenotype_file=pheno,
        info=INFO,
        save_kinship=True,
        output_dir=tmp_path,
        check_memory=False,
        show_progress=False,
    )
    keep = _reference_keep(bgen_files, _analysed_rows(), info_threshold=INFO)
    assert result.n_snps_tested == int(keep.sum())
    dosages = decode_file(bgen_files.bgen).dosages
    np.testing.assert_allclose(
        read_kinship_matrix(tmp_path / "result.cXX.npy", n_samples=N_SAMPLES),
        compute_centered_kinship(dosages[:, keep], check_memory=False),
        rtol=1e-10,
        atol=1e-14,
    )


def test_counted_allele_is_the_first_bgen_allele(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path
):
    out = tmp_path / "out"
    base = [*_bgen_args(bgen_files, pheno), "-outdir", str(out)]
    _invoke(["-lmm", "1", "-loco", *base, "-o", "assoc"])

    rows = _analysed_rows()
    dosages = decode_file(bgen_files.bgen).dosages[rows, :]
    results = list(load_gemma_assoc(out / "assoc.assoc.txt"))
    assert results
    for r in results:
        j = int(r.rs.removeprefix("rs"))
        assert (r.allele1, r.allele0) == ("C", "T")
        assert abs(r.af - np.nanmean(dosages[:, j]) / 2) <= 5e-4 + 1e-12


def test_loco_with_info_tests_the_reference_snp_set(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path
):
    out = tmp_path / "out"
    base = [*_bgen_args(bgen_files, pheno), "-info", str(INFO), "-outdir", str(out)]
    _invoke(["-lmm", "1", "-loco", *base, "-o", "assoc"])

    keep = _reference_keep(bgen_files, _analysed_rows(), info_threshold=INFO)
    rs = np.array([f"rs{j}" for j in range(N_VARIANTS)])
    assert sorted(_tested(out / "assoc.assoc.txt")) == sorted(rs[keep])


def test_loco_kinship_uses_the_info_filter(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path
):
    out = tmp_path / "out"
    base = [*_bgen_args(bgen_files, pheno), "-info", str(INFO), "-outdir", str(out)]
    _invoke(["-gk", "1", "-loco", *base, "-o", "kin"])

    keep = _reference_keep(bgen_files, _analysed_rows(), info_threshold=INFO)
    dosages = decode_file(bgen_files.bgen).dosages
    chromosomes = np.array(CHROMOSOMES)
    for chr_name in ("1", "2", "3"):
        expected = compute_centered_kinship(
            dosages[:, keep & (chromosomes != chr_name)], check_memory=False
        )
        np.testing.assert_allclose(
            read_kinship_matrix(
                out / f"kin.loco.cXX.chr{chr_name}.npy", n_samples=N_SAMPLES
            ),
            expected,
            rtol=1e-10,
            atol=1e-14,
        )


def test_lmm_loco_kinship_uses_the_info_filter(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path
):
    """Each chromosome's eigenvalues are those of its reference K_loco.

    ``-n 2`` has no missing value, so the analysed rows are every row and
    analysis centering leaves the reference matrix unchanged.
    """
    eigen_dir = tmp_path / "eigen"
    args = [*_bgen_args(bgen_files, pheno), "-n", "2", "-info", str(INFO)]
    args += ["-eigen", "--eigen-dir", str(eigen_dir), "-outdir", str(tmp_path)]
    _invoke(["-lmm", "1", "-loco", *args])

    keep = _reference_keep(bgen_files, np.arange(N_SAMPLES), info_threshold=INFO)
    dosages = decode_file(bgen_files.bgen).dosages
    chromosomes = np.array(CHROMOSOMES)
    manifest = json.loads(eigen_cache_manifest_path(eigen_dir, "result").read_text())
    members = resolve_eigen_cache(manifest, eigen_dir, "result", ["1", "2", "3"])
    assert members is not None
    for chr_name, (d_path, _) in members.items():
        reference = compute_centered_kinship(
            dosages[:, keep & (chromosomes != chr_name)], check_memory=False
        )
        np.testing.assert_allclose(
            np.sort(np.load(d_path)),
            np.sort(np.clip(np.linalg.eigvalsh(reference), 0, None)),
            rtol=1e-8,
            atol=1e-10,
        )


def test_sample_and_bgi_default_from_the_bgen_path(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path
):
    """Moved elsewhere, the defaults miss and the explicit flags find them."""
    moved = tmp_path / "moved"
    moved.mkdir()
    sample = Path(shutil.move(bgen_files.sample, moved / "other.sample"))
    bgi = Path(shutil.move(bgen_files.bgi, moved / "other.bgi"))
    args = ["-gk", "1", *_bgen_args(bgen_files, pheno), "-outdir", str(tmp_path)]

    result = runner.invoke(main, args)
    assert result.exit_code == 1
    assert f"BGEN .sample file not found: {bgen_files.sample}" in result.output

    _invoke([*args, "-sample", str(sample), "-bgi", str(bgi)])


# --------------------------------------------------------------------------
# -p phenotype file
# --------------------------------------------------------------------------


def _fam_as_phenotype_file(bfile: Path, path: Path) -> Path:
    """The ``.fam`` phenotype columns (6 onward) as a GEMMA ``-p`` file."""
    rows = [
        line.split()[5:]
        for line in bfile.with_suffix(".fam").read_text().split("\n")
        if line
    ]
    path.write_text("".join("\t".join(r) + "\n" for r in rows))
    return path


def test_p_with_bfile_matches_the_fam_column(tmp_path: Path):
    """The same column through ``-p`` gives byte-identical kinship and results."""
    bfile = SYNTHETIC.bfile
    require_fixture(*(bfile.with_suffix(ext) for ext in (".bed", ".bim", ".fam")))
    pheno = _fam_as_phenotype_file(bfile, tmp_path / "pheno.txt")
    outputs = {}
    for label, extra in (("fam", []), ("p", ["-p", str(pheno)])):
        out = tmp_path / label
        base = ["-bfile", str(bfile), *extra, "-outdir", str(out)]
        _invoke(["-gk", "1", *base, "-o", "kin"])
        _invoke(["-lmm", "4", *base, "-k", str(out / "kin.cXX.npy"), "-o", "a"])
        outputs[label] = (
            (out / "kin.cXX.npy").read_bytes(),
            (out / "a.assoc.txt").read_bytes(),
        )
    assert outputs["p"] == outputs["fam"]


def test_p_row_count_must_match_the_dataset(bgen_files: BgenFiles, tmp_path: Path):
    lines = _write_phenotypes(tmp_path / "p.txt", ()).read_text().splitlines()
    short = tmp_path / "short.txt"
    short.write_text("\n".join(lines[:-1]) + "\n")
    result = runner.invoke(
        main, ["-gk", "1", *_bgen_args(bgen_files, short), "-outdir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert (
        f"Phenotype file {short} has {N_SAMPLES - 1} rows but the genotype data "
        f"has {N_SAMPLES} samples" in result.output
    )


def test_covariate_row_count_is_checked_against_bgen(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path
):
    covariates = tmp_path / "cov.txt"
    covariates.write_text("1\n" * (N_SAMPLES + 1))
    result = runner.invoke(
        main,
        [
            "-gk",
            "1",
            *_bgen_args(bgen_files, pheno),
            "-c",
            str(covariates),
            "-outdir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert f"{N_SAMPLES + 1} rows but the genotype data has {N_SAMPLES}" in (
        result.output
    )


def test_p_selects_the_n_column(bgen_files: BgenFiles, pheno: Path, tmp_path: Path):
    """``-n 2`` reads column 2, which has no missing values."""
    out = tmp_path / "out"
    base = [*_bgen_args(bgen_files, pheno), "-outdir", str(out), "-loco"]
    _invoke(["-lmm", "1", *base, "-n", "2", "-o", "assoc"])
    keep = _reference_keep(bgen_files, np.arange(N_SAMPLES), info_threshold=0)
    assert len(_tested(out / "assoc.assoc.txt")) == int(keep.sum())


# --------------------------------------------------------------------------
# Rejections at the config boundary
# --------------------------------------------------------------------------


REJECTIONS = [
    (
        ["-bgen", "x.bgen", "-p", "p.txt", "-hwe", "0.001"],
        "-hwe is not supported with -bgen: HWE genotype classes need hard "
        "calls, and -bgen holds probabilities",
    ),
    (
        ["-bfile", "x", "-info", "0.8"],
        "-info is not supported with -bfile: INFO needs genotype "
        "probabilities, and -bfile holds hard calls",
    ),
    (
        ["-bgen", "x.bgen", "-p", "p.txt", "--backend", "numpy"],
        "--backend numpy is not supported with -bgen: the batch runner loads "
        "hard calls into memory, and -bgen holds probabilities. Use --backend "
        "numpy-streaming or auto.",
    ),
    (
        ["-bgen", "x.bgen"],
        "-bgen requires -p (phenotype file): -bgen input carries no phenotypes",
    ),
    (["-bfile", "x", "-bgen", "x.bgen"], "Exactly one of -bfile or -bgen is required"),
    ([], "Exactly one of -bfile or -bgen is required"),
    (
        ["-bfile", "x", "-sample", "x.sample"],
        "-sample and -bgi apply only to -bgen input",
    ),
    (["-bgen", "x.bgen", "-p", "p.txt", "-info", "-1"], "info_threshold must be >= 0"),
]


@pytest.mark.parametrize(("args", "message"), REJECTIONS)
def test_cli_rejects_before_reading_any_file(args: list[str], message: str):
    result = runner.invoke(main, ["-lmm", "1", "-k", "k.txt", *args])
    assert result.exit_code == 2
    assert message in " ".join(result.output.split())


def test_input_encoding_matches_the_opened_dataset(bgen_files: BgenFiles):
    """The config rules read the input's encoding before anything is opened."""
    require_fixture(
        *(SYNTHETIC.bfile.with_suffix(ext) for ext in (".bed", ".bim", ".fam"))
    )
    for config in (
        PipelineConfig(bfile=SYNTHETIC.bfile),
        PipelineConfig(bgen=bgen_files.bgen, phenotype_file=Path("p")),
    ):
        genotypes = config.genotypes()
        assert genotypes.encoding is genotypes.open().encoding


def test_backend_numpy_with_bgen_is_accepted_under_loco():
    """LOCO never runs the batch runner, so the backend request is moot."""
    PipelineConfig(
        bgen=Path("x.bgen"), phenotype_file=Path("p"), loco=True, backend="numpy"
    )


def test_env_backend_numpy_with_bgen_fails_before_any_pass(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("JAMMA_BACKEND", "numpy")
    config = PipelineConfig(
        bgen=bgen_files.bgen,
        phenotype_file=pheno,
        output_dir=tmp_path,
        check_memory=False,
        show_progress=False,
    )
    with pytest.raises(
        ValueError, match="JAMMA_BACKEND=numpy is not supported with -bgen"
    ):
        PipelineRunner(config).run()


# --------------------------------------------------------------------------
# LOCO eigen cache over BGEN
# --------------------------------------------------------------------------


def test_bgen_loco_eigen_cache_cold_write_warm(
    bgen_files: BgenFiles, pheno: Path, tmp_path: Path
):
    """No cache computes; ``write_eigen`` commits a BGEN key; the next run hits it."""
    eigen_dir = tmp_path / "eigen"
    eigen_dir.mkdir()

    def run(label: str, *, write_eigen: bool = False) -> tuple[bytes, list[str]]:
        messages: list[str] = []
        handler = logger.add(messages.append, level="INFO", format="{message}")
        try:
            gwas(
                bgen=bgen_files.bgen,
                phenotype_file=pheno,
                info=INFO,
                loco=True,
                eigen_dir=eigen_dir,
                output_dir=tmp_path / label,
                check_memory=False,
                show_progress=False,
                write_eigen=write_eigen,
            )
        finally:
            logger.remove(handler)
        return (tmp_path / label / "result.assoc.txt").read_bytes(), messages

    hit = "Found complete LOCO eigen cache"
    manifest = eigen_cache_manifest_path(eigen_dir, "result")

    cold, messages = run("cold")
    assert not any(hit in m for m in messages)
    assert not manifest.exists()

    written, messages = run("write", write_eigen=True)
    assert not any(hit in m for m in messages)
    components = json.loads(manifest.read_text())["components"]
    assert {"bgen_fingerprint", "sample_sha256", "variants_sha256"} <= set(components)
    assert "bed_fingerprint" not in components
    assert components["info_threshold"] == INFO

    warm, messages = run("warm")
    assert any(hit in m for m in messages)
    assert warm == written == cold
