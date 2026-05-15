"""Smoke tests for the Streamlit demo's helper module.

Exercises the data path (fixture discovery, Bundle parsing, summary
building, ablation file lookup) without launching Streamlit — the
helpers in `app/lib.py` import only `pgx_digest`, so this runs in the
library's normal pytest environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "app"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(APP_DIR))

import lib as app_lib  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


def test_find_fixtures_returns_sorted_json_paths() -> None:
    fixtures = app_lib.find_fixtures(FIXTURES_DIR)
    assert fixtures, "expected at least one fixture under tests/fixtures/"
    assert all(p.suffix == ".json" for p in fixtures)
    assert fixtures == sorted(fixtures)


def test_find_fixtures_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert app_lib.find_fixtures(tmp_path / "does-not-exist") == []


@pytest.mark.parametrize(
    "fixture_name",
    [
        "pharmcat_cyp2c19_minimal.json",
        "pharmcat_multigene.json",
    ],
)
def test_summarize_bundle_on_real_fixtures(fixture_name: str) -> None:
    bundle = app_lib.parse_pharmcat_json(FIXTURES_DIR / fixture_name)
    summary = app_lib.summarize_bundle(bundle)
    assert summary.n_genes == len(bundle.items)
    assert summary.n_drugs == sum(
        len(f.affected_drugs) for f in bundle.items
    )
    assert len(summary.rows) == summary.n_genes
    for row in summary.rows:
        assert {"Gene", "Diplotype", "Phenotype", "Drugs", "# Drugs"} <= row.keys()


def test_find_ablation_files_missing_dir(tmp_path: Path) -> None:
    assert app_lib.find_ablation_files(tmp_path / "no-eval-results") == []


def test_find_ablation_files_sorted(tmp_path: Path) -> None:
    run_a = tmp_path / "20260101_aaa_ablations"
    run_b = tmp_path / "20260102_bbb_ablations"
    run_a.mkdir()
    run_b.mkdir()
    (run_b / "ablation_b_model.md").write_text("# B")
    (run_a / "ablation_a_verifier.md").write_text("# A")
    found = app_lib.find_ablation_files(tmp_path)
    assert [p.name for p in found] == [
        "ablation_a_verifier.md",
        "ablation_b_model.md",
    ]
