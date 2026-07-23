"""Headless AppTest coverage for the Streamlit demo (M6).

streamlit.testing.v1.AppTest runs app.py the way `streamlit run` would --
minus a browser -- so these are smoke tests over the real, committed
results/ artifacts: they catch "the app throws" regressions (schema drift,
a stale cache_data signature, ...) that the pure-function tests elsewhere in
this suite can't see, since app.py itself is otherwise never exercised.

The guard tests clone the app + a mutable results/ tree into a temp dir
(AppTest sets __file__ to the copied path, so app.py's ROOT relocates
there) and then delete or corrupt one artifact. Each asserts the app
renders one friendly German error box -- a missing file, a present-but-
wrong-schema CSV, or malformed JSON -- instead of a raw traceback.
"""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "app.py"


@pytest.fixture(autouse=True)
def _isolate_streamlit_cache():
    # st.cache_data is a process-global cache keyed by function code + args,
    # not by AppTest instance. Without this, a loader cached by one test
    # (e.g. a valid learning_curve.csv from the default-load test) would be
    # served to a later test's cloned app -- masking the corrupt-artifact
    # guards, which only fire when the loader actually re-reads the file.
    st.cache_data.clear()
    yield


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


def _clone_app(tmp_root: Path, *, with_data: bool = False) -> Path:
    """Copy app.py + a mutable results/ tree into tmp_root so a test can
    delete or corrupt artifacts without touching the real repo. AppTest sets
    __file__ to the returned path, so the app's ROOT (hence RESULTS_DIR /
    DATA_DIR) relocates here. data/ is symlinked -- it is read-only, and only
    the live-refit path (the malformed-JSON guard) ever reaches it."""
    shutil.copy(APP_PATH, tmp_root / "app.py")
    shutil.copytree(REPO_ROOT / "results", tmp_root / "results")
    if with_data:
        (tmp_root / "data").symlink_to(REPO_ROOT / "data")
    return tmp_root / "app.py"


def test_app_runs_without_exception_on_default_load():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=120)

    assert not at.exception
    assert at.title[0].value == "Tabular Showdown: TabPFN v2 vs. tuned LightGBM"
    assert len(at.dataframe) == 1  # the "Metrics at a glance" table


def test_app_renders_the_per_size_verdict_table_with_the_n5000_tie():
    # The verdict table is rendered as st.markdown (the same string the README
    # block carries), so it shows up in at.markdown, not at.dataframe.
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=120)
    assert not at.exception

    blob = "\n".join(m.value for m in at.markdown)
    assert "Mean ROC-AUC diff (TabPFN − LightGBM)" in blob
    assert "| 5,000 | 5 |" in blob
    assert "Wilcoxon p=0.625 † | tie |" in blob  # the n=5000 tie verdict, same as README
    assert "| 2,000 | 10 | +0.0033 |" in blob
    assert "| TabPFN wins |" in blob


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
        app = _clone_app(tmp_root)
        (tmp_root / missing_rel_path).unlink()

        at = AppTest.from_file(str(app))
        at.run(timeout=60)

    assert not at.exception  # a friendly stop(), not a raw traceback
    assert len(at.error) == 1
    message = at.error[0].value
    assert "fehlen" in message  # German error box, not the old English wording
    assert missing_rel_path in message  # names the file that is missing
    assert f"uv run python {script}" in message  # names the runnable fix
    # The up-front guard halts before anything downstream renders -- in
    # particular no st.image() has been reached (that is the traceback this
    # guard prevents).
    assert len(at.title) == 0
    assert len(at.image) == 0


# --- Malformed-content guards: present-but-corrupt artifacts ------------------


def test_app_shows_german_guard_when_curve_csv_has_wrong_schema():
    # Unlike the up-front missing-file guard, _load_curve_csv's schema check
    # fires during normal render -- after the title and the money-chart image
    # -- so this asserts the error box, not their absence (they render first).
    with TemporaryDirectory(prefix="tabular-showdown-test-app-") as tmp:
        tmp_root = Path(tmp)
        app = _clone_app(tmp_root)
        # Valid CSV, but none of the columns the app needs.
        (tmp_root / "results" / "learning_curve.csv").write_text("a,b\n1,2\n")

        at = AppTest.from_file(str(app))
        at.run(timeout=60)

    assert not at.exception
    assert len(at.error) == 1
    message = at.error[0].value
    assert "Spalten" in message  # German column-schema wording
    assert "results/learning_curve.csv" in message
    assert "uv run python scripts/run_curve.py" in message


@pytest.mark.parametrize(
    "content", ["", "a,b,c\n1,2\n3,4,5,6\n"], ids=["empty-file", "ragged-rows"]
)
def test_app_shows_german_guard_when_curve_csv_is_unparseable(content):
    # pd.read_csv itself raises before the schema check ever runs --
    # EmptyDataError on an empty file, ParserError on ragged rows -- so this
    # exercises the read guard, not the column check. Like the schema case it
    # fires mid-render (after the title + money-chart image render).
    with TemporaryDirectory(prefix="tabular-showdown-test-app-") as tmp:
        tmp_root = Path(tmp)
        app = _clone_app(tmp_root)
        (tmp_root / "results" / "learning_curve.csv").write_text(content)

        at = AppTest.from_file(str(app))
        at.run(timeout=60)

    assert not at.exception
    assert len(at.error) == 1
    message = at.error[0].value
    assert "beschädigt" in message  # German "corrupted" wording
    assert "results/learning_curve.csv" in message
    assert "uv run python scripts/run_curve.py" in message


def test_app_shows_german_guard_when_lgbm_params_json_is_malformed():
    # _load_lgbm_params is reached only through the live-refit button (it
    # feeds _live_refit), so this drives the button; the guard fires on the
    # bad JSON before any model is fit. data/ is required because _live_refit
    # loads the frozen eval set before it ever reads the params.
    with TemporaryDirectory(prefix="tabular-showdown-test-app-") as tmp:
        tmp_root = Path(tmp)
        app = _clone_app(tmp_root, with_data=True)
        (tmp_root / "results" / "lgbm_baseline.json").write_text("{ not valid json")

        at = AppTest.from_file(str(app))
        at.run(timeout=60)
        assert not at.exception  # normal render is fine; params not read yet

        at.slider[0].set_value(100)  # live n_train; guard fires before any fit
        at.button[0].set_value(True)
        at.run(timeout=120)

    assert not at.exception
    assert len(at.error) == 1
    message = at.error[0].value
    assert "beschädigt" in message  # German "corrupted" wording
    assert "results/lgbm_baseline.json" in message
    assert "uv run python scripts/run_lgbm_baseline.py" in message


def test_app_shows_german_guard_when_lgbm_params_json_lacks_params_key():
    # Valid JSON, but no "params" key -> KeyError, the other half of
    # _load_lgbm_params' guard (its docstring claims both). Reached only via
    # the live-refit button, exactly like the malformed-JSON case above.
    with TemporaryDirectory(prefix="tabular-showdown-test-app-") as tmp:
        tmp_root = Path(tmp)
        app = _clone_app(tmp_root, with_data=True)
        (tmp_root / "results" / "lgbm_baseline.json").write_text("{}")

        at = AppTest.from_file(str(app))
        at.run(timeout=60)
        assert not at.exception

        at.slider[0].set_value(100)
        at.button[0].set_value(True)
        at.run(timeout=120)

    assert not at.exception
    assert len(at.error) == 1
    message = at.error[0].value
    assert "beschädigt" in message
    assert "results/lgbm_baseline.json" in message
    assert "uv run python scripts/run_lgbm_baseline.py" in message
