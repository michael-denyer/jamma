"""Protect the benchmark workload and fail closed on incomplete results."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0


def _load_script(name):
    scripts = str(Path(__file__).resolve().parent.parent / "scripts")
    sys.path.insert(0, scripts)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(scripts)


common = _load_script("_bench_common")
backends = _load_script("bench_all_backends")
loco = _load_script("bench_loco")


def test_full_gwas_keeps_jamma_kinship_in_memory(tmp_path):
    """Only final association output is required; GEMMA needs two processes."""
    gemma = backends.commands_for(["gemma"], "gwas_wald", tmp_path, None)
    jamma = backends.commands_for(
        [sys.executable, "-m", "jamma"], "gwas_wald", tmp_path, "numpy"
    )
    assert len(gemma) == 2
    assert "-gk" in gemma[0]
    assert gemma[1][gemma[1].index("-k") + 1] == str(tmp_path / "bench.cXX.txt")
    assert len(jamma) == 1
    assert "gwas(" in jamma[0][2]
    assert "save_kinship=True" not in jamma[0][2]


def test_standalone_kinship_requires_same_text_artifact(tmp_path):
    command = backends.commands_for(
        [sys.executable, "-m", "jamma"], "kinship", tmp_path, "numpy"
    )[0]
    assert "--legacy-text" in command
    assert "-gk" in command


def test_loco_computes_excluded_kinship_then_tests_only_held_out_snps(tmp_path):
    snps = tmp_path / "chr1.snps"
    commands = loco.gemma_commands(Path("gemma"), tmp_path, {"1": snps}, True)
    kinship, association = commands
    assert kinship[kinship.index("-snps") + 1] == str(snps.with_suffix(".kinship.snps"))
    assert "-k" not in kinship
    assert "-loco" not in kinship
    assert association[association.index("-k") + 1] == str(tmp_path / "chr1.cXX.txt")
    assert association[association.index("-snps") + 1] == str(snps)
    assert all("-c" in command for command in commands)


def test_process_failure_cannot_produce_a_benchmark_time():
    with pytest.raises(RuntimeError, match="Benchmark command failed"):
        common.run_commands(
            [[sys.executable, "-c", "raise SystemExit(7)"]], dict(os.environ)
        )


def test_process_timer_waits_for_final_output(tmp_path):
    output = tmp_path / "finished"
    elapsed = common.run_commands(
        [
            [
                sys.executable,
                "-c",
                "import sys,time; from pathlib import Path; "
                "time.sleep(0.05); Path(sys.argv[1]).write_text('done')",
                str(output),
            ]
        ],
        dict(os.environ),
    )
    assert elapsed >= 0.05
    assert output.read_text() == "done"


def test_output_validation_rejects_missing_or_repeated_snps(tmp_path):
    path = tmp_path / "results.assoc.txt"
    path.write_text("rs\tallele1\tallele0\tbeta\tse\tp_wald\nrs1\tA\tC\t1\t1\t0.1\n")
    rows = common.read_associations(path)
    with pytest.raises(ValueError, match="SNP sets differ"):
        common.verify_associations(rows, {})
    path.write_text(path.read_text() + "rs1\tA\tC\t1\t1\t0.1\n")
    with pytest.raises(ValueError, match="duplicate"):
        common.read_associations(path)


def test_loco_snp_lists_are_disjoint_and_cover_the_input(tmp_path, monkeypatch):
    prefix = tmp_path / "study"
    prefix.with_suffix(".bim").write_text(
        "1 rs1 0 10 A C\n1 rs2 0 20 A C\n2 rs3 0 30 A C\n"
    )
    monkeypatch.setattr(loco, "MOUSE_PREFIX", prefix)
    lists = loco.prepare_inputs(tmp_path)
    for chrom, path in lists.items():
        tested = set(path.read_text().splitlines())
        kinship = set(path.with_suffix(".kinship.snps").read_text().splitlines())
        assert not tested & kinship
        assert tested | kinship == {"rs1", "rs2", "rs3"}
        assert tested == ({"rs1", "rs2"} if chrom == "1" else {"rs3"})


def test_output_validation_accepts_round_off_on_a_null_effect():
    row = {"allele1": "A", "allele0": "C", "se": "3.515983e-02", "p_wald": "0.9999"}
    common.verify_associations(
        {"rs13475789": {**row, "beta": "4.366448e-06"}},
        {"rs13475789": {**row, "beta": "4.444851e-06"}},
    )


def test_output_validation_rejects_numerical_disagreement():
    row = {"allele1": "A", "allele0": "C", "beta": "1", "se": "1", "p_wald": "0.1"}
    with pytest.raises(AssertionError, match="beta"):
        common.verify_associations({"rs1": row}, {"rs1": {**row, "beta": "2"}})
