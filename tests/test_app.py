"""Headless AppTest coverage for the Streamlit demo (M6).

streamlit.testing.v1.AppTest runs app.py the way `streamlit run` would --
minus a browser -- so these are smoke tests over the real, committed
results/ artifacts: they catch "the app throws" regressions (schema drift,
a stale cache_data signature, ...) that the pure-function tests elsewhere in
this suite can't see, since app.py itself is otherwise never exercised.

The missing-artifact tests copy the app + a results/ tree into a temp dir
(AppTest sets __file__ to the copied path, so app.py's ROOT relocates
there), delete one required artifact, and assert the app renders one
friendly German error box -- not a raw traceback from the first st.image().
"""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "app.py"

# Contract mirror of app.REQUIRED_RESULT_FILES: every results/ artifact the
# app hard-requires, paired with the script whose `uv run python` invocation
# (re)produces it. Kept here rather than imported because importing app.py
# executes its whole top-level render.
REQUIRED_ARTIFACTS = [
    ("results/learning_curve.csv", "scripts/run_curve.py"),
    ("results/learning_curve.png", "scripts/run_curve.py"),
    ("results/lgbm_baseline.json", "scripts/run_lgbm_baseline.py"),
    ("results/figures/calibration.png", "scripts/make_figures.py"),
    ("results/figures/timing.png", "scripts/make_figures.py"),
    ("results/figures/shap_summary.png", "scripts/make_figures.py"),
    ("results/figures/permutation_importance.png", "scripts/make_figures.py"),
]


def test_app_runs_without_exception_on_default_load():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=120)

    assert not at.exception
    assert at.title[0].value == "Tabular Showdown: TabPFN v2 vs. tuned LightGBM"
    assert len(at.dataframe) == 1  # the "Metrics at a glance" table


def test_app_size_slider_updates_the_metrics_table_without_exception():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=120)
    assert not at.exception

    at.select_slider[0].set_value(200)  # smallest size on the curve
    at.run(timeout=120)

    assert not at.exception
    assert len(at.dataframe) == 1


def test_app_live_refit_button_fits_logreg_and_lgbm_at_smallest_size():
    # Exercises _live_refit -- the one path that trains live rather than
    # reading a precomputed artifact. Pinned to the slider's minimum (100
    # rows, its smallest allowed value) to keep this fast and CI-safe:
    # LogReg + a 10-trial Optuna LightGBM search both fit in a couple of
    # seconds at this size, unlike TabPFN which this slider deliberately
    # excludes (see app.py's module docstring).
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=120)
    assert not at.exception

    at.slider[0].set_value(100)
    at.button[0].set_value(True)
    at.run(timeout=180)

    assert not at.exception
    assert len(at.dataframe) == 2  # the size-slider table + the live-refit table


# --- Missing-artifact guard: one friendly German box, never a traceback -------


@pytest.mark.parametrize(
    "missing_rel_path, script", REQUIRED_ARTIFACTS, ids=[a[0] for a in REQUIRED_ARTIFACTS]
)
def test_app_shows_german_guard_when_a_required_artifact_is_missing(missing_rel_path, script):
    with TemporaryDirectory(prefix="tabular-showdown-test-app-") as tmp:
        tmp_root = Path(tmp)
        shutil.copy(APP_PATH, tmp_root / "app.py")
        shutil.copytree(REPO_ROOT / "results", tmp_root / "results")
        (tmp_root / missing_rel_path).unlink()

        at = AppTest.from_file(str(tmp_root / "app.py"))
        at.run(timeout=60)

    assert not at.exception  # a friendly stop(), not a raw traceback
    assert len(at.error) == 1
    message = at.error[0].value
    assert "fehlen" in message  # German error box, not the old English wording
    assert missing_rel_path in message  # names the file that is missing
    assert f"uv run python {script}" in message  # names the runnable fix
    # The guard runs up front, so nothing downstream renders -- in particular
    # no st.image() has been reached (that is the traceback this guard prevents).
    assert len(at.title) == 0
    assert len(at.image) == 0
