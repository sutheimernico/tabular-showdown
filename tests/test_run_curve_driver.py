"""Tests for scripts/run_curve.py wiring that lives outside the src package.

Focus: the TabPFN compute-valve override (--tabpfn-time-valve-seconds). The
in-run valve itself is covered in test_curve.py; here we pin the resume-time
half -- _load_skip_keys reconstructing (or, when overridden, NOT reconstructing)
the "skip larger TabPFN sizes" decision from persisted timings.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DRIVER_PATH = ROOT / "scripts" / "run_curve.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("run_curve_driver", DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_slow_tabpfn_csv(path: Path) -> None:
    """A CSV whose only TabPFN point (n=5000) blew past the default budget."""
    pd.DataFrame(
        {
            "model": ["tabpfn"],
            "n_train": [5000],
            "seed": [0],
            "roc_auc": [0.914],
            "accuracy": [0.855],
            "log_loss": [0.303],
            "brier": [0.097],
            "fit_s": [0.12],
            "predict_s": [755.0],
        }
    ).to_csv(path, index=False)


def test_default_valve_reconstructs_skip_for_larger_tabpfn_size(driver, tmp_path, monkeypatch):
    csv = tmp_path / "learning_curve.csv"
    _write_slow_tabpfn_csv(csv)
    monkeypatch.setattr(driver, "CSV_PATH", csv)

    skip, _ = driver._load_skip_keys(driver.TABPFN_MAX_SECONDS)

    # 5000 exceeded the 480s default -> every larger TabPFN size is pre-skipped.
    assert ("tabpfn", 10000, 0) in skip


def test_raised_valve_disables_the_skip_reconstruction(driver, tmp_path, monkeypatch):
    csv = tmp_path / "learning_curve.csv"
    _write_slow_tabpfn_csv(csv)
    monkeypatch.setattr(driver, "CSV_PATH", csv)

    # Budget above the point's own cost -> nothing is "slow", no larger size skipped.
    skip, _ = driver._load_skip_keys(1e9)

    assert ("tabpfn", 10000, 0) not in skip
