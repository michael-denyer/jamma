"""Behaviour tests for ``scripts/kinship_digest.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import require_fixture
from tests.fixture_paths import SYNTHETIC

pytestmark = pytest.mark.tier1

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "kinship_digest.py"


@pytest.fixture
def digest_module(monkeypatch):
    # Restrict to gemma_synthetic; the full mouse_hs1940 sweep is the
    # live-lane's job, not a unit test's.
    spec = importlib.util.spec_from_file_location("kinship_digest", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["kinship_digest"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FIXTURES", {"gemma_synthetic": SYNTHETIC.bfile})
    monkeypatch.setattr(module, "LOCO_FIXTURES", set())
    yield module
    del sys.modules["kinship_digest"]


def test_two_runs_on_the_same_fixture_agree_exactly(digest_module, tmp_path):
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    assert digest_module.main(["--out", str(out_a)]) == 0
    assert digest_module.main(["--out", str(out_b)]) == 0

    assert digest_module.main(["--diff", str(out_a), str(out_b)]) == 0


def test_diff_of_a_file_against_itself_reports_zero_and_exits_zero(
    digest_module, capsys, tmp_path
):
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)

    out = tmp_path / "k.json"
    digest_module.main(["--out", str(out)])
    capsys.readouterr()

    assert digest_module.main(["--diff", str(out), str(out)]) == 0
    assert "0 keys differ" in capsys.readouterr().out


def test_a_perturbed_matrix_names_a_gemma_synthetic_key(
    digest_module, monkeypatch, capsys, tmp_path
):
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)

    baseline = tmp_path / "baseline.json"
    assert digest_module.main(["--out", str(baseline)]) == 0
    capsys.readouterr()

    real_loader = digest_module.load_plink_binary

    def _perturbed_loader(bfile):
        data = real_loader(bfile)
        data.genotypes[0, 0] = (data.genotypes[0, 0] + 1.0) % 3.0
        return data

    monkeypatch.setattr(digest_module, "load_plink_binary", _perturbed_loader)

    perturbed = tmp_path / "perturbed.json"
    assert digest_module.main(["--out", str(perturbed)]) == 0

    exit_code = digest_module.main(["--diff", str(baseline), str(perturbed)])
    err = capsys.readouterr().err

    assert exit_code == 1
    assert "gemma_synthetic" in err
    assert "differ" in err


def test_header_mismatch_on_backend_refuses_to_compare(digest_module, tmp_path):
    require_fixture(SYNTHETIC.bed, SYNTHETIC.bim, SYNTHETIC.fam)

    out = tmp_path / "k.json"
    digest_module.main(["--out", str(out)])

    data = json.loads(out.read_text())
    data["header"]["blas_backend"] = "SomeOtherBackend"
    tampered = tmp_path / "k_other_backend.json"
    tampered.write_text(json.dumps(data))

    assert digest_module.main(["--diff", str(out), str(tampered)]) == 2


def test_digest_array_shape_prefix_prevents_reshape_collision(digest_module):
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    b = np.arange(6, dtype=np.float64).reshape(3, 2)
    assert digest_module.digest_array(a) != digest_module.digest_array(b)
