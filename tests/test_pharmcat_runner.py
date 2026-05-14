"""Tests for the PharmCAT Docker runner.

Most tests gate on `docker_available()` and skip if Docker isn't usable
in the current environment. CI without Docker still passes; CI with
Docker exercises a real end-to-end run on a synthetic fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pgx_digest.pharmcat_runner import (
    DockerUnavailable,
    PharmCATRunError,
    _vcf_basename,
    docker_available,
    run_pharmcat,
    vcf_to_bundle,
)

FIXTURES = Path(__file__).parent / "fixtures"
NEEDS_DOCKER = pytest.mark.skipif(
    not docker_available(),
    reason="docker not available in this environment",
)


def test_vcf_basename_strips_plain_vcf() -> None:
    assert _vcf_basename(Path("foo.vcf")) == "foo"


def test_vcf_basename_strips_vcf_gz() -> None:
    assert _vcf_basename(Path("foo.vcf.gz")) == "foo"


def test_vcf_basename_strips_vcf_bgz() -> None:
    assert _vcf_basename(Path("sample-1.vcf.bgz")) == "sample-1"


def test_vcf_basename_preserves_dots_in_stem() -> None:
    assert _vcf_basename(Path("NA12878.GRCh38.vcf")) == "NA12878.GRCh38"


def test_run_pharmcat_raises_on_missing_vcf(tmp_path: Path) -> None:
    bogus = tmp_path / "nonexistent.vcf"
    with pytest.raises(FileNotFoundError):
        run_pharmcat(bogus, output_dir=tmp_path)


def test_run_pharmcat_raises_docker_unavailable_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Force the docker_available check to return False and verify raise."""
    monkeypatch.setattr(
        "pgx_digest.pharmcat_runner.docker_available", lambda: False
    )
    vcf = FIXTURES / "cyp2c19_star1_star2_het.vcf"
    with pytest.raises(DockerUnavailable):
        run_pharmcat(vcf, output_dir=tmp_path)


@NEEDS_DOCKER
def test_run_pharmcat_smoke(tmp_path: Path) -> None:
    """Live run against a synthetic fixture. Validates wiring only.

    The synthetic VCF only covers one position, so PharmCAT will warn
    about missing positions and may not produce a complete diplotype
    call — but it should produce *a* report.json without crashing.
    """
    vcf = FIXTURES / "cyp2c19_star1_star2_het.vcf"
    report = run_pharmcat(vcf, output_dir=tmp_path, timeout=120)
    assert report.exists()
    assert report.name.endswith(".report.json")


@NEEDS_DOCKER
def test_vcf_to_bundle_smoke(tmp_path: Path) -> None:
    """End-to-end: VCF -> Docker PharmCAT -> parsed Bundle."""
    vcf = FIXTURES / "cyp2c19_star1_star2_het.vcf"
    bundle = vcf_to_bundle(vcf, output_dir=tmp_path, timeout=120)
    assert bundle.source.endswith(".report.json")
    assert "pharmcat_version" in bundle.metadata
