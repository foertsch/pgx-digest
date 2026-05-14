"""Run PharmCAT via Docker and feed the result into the parser.

Wraps ``pgkb/pharmcat``'s ``pharmcat_pipeline`` script, which performs
the full chain:

    VCF -> preprocessor -> matcher -> phenotyper -> reporter -> JSON

We invoke the container with two bind mounts (input directory and
output directory) and pick up the ``*.report.json`` file PharmCAT
emits, which is what ``pgx_digest.pharmcat.parse_pharmcat_json``
consumes.

Requirements:
- Docker installed and the daemon running
- First invocation pulls ``pgkb/pharmcat`` (~1 GB); subsequent runs reuse
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Final

from pgx_digest.bundle import Bundle, PGxFinding, PrivacyTier
from pgx_digest.pharmcat import parse_pharmcat_json

DEFAULT_IMAGE: Final[str] = "pgkb/pharmcat"
DEFAULT_TIMEOUT_S: Final[int] = 1800  # 30 min — generous for WGS VCFs


class DockerUnavailable(RuntimeError):
    """Docker is not installed or the daemon is not running."""


class PharmCATRunError(RuntimeError):
    """The PharmCAT pipeline failed or produced no JSON report."""


def docker_available() -> bool:
    """Return True if `docker` is on PATH and the daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _vcf_basename(vcf_path: Path) -> str:
    """Strip .vcf, .vcf.gz, .vcf.bgz to get PharmCAT's output basename."""
    name = vcf_path.name
    for suffix in (".gz", ".bgz"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.endswith(".vcf"):
        name = name[: -len(".vcf")]
    return name


def _invoke_pharmcat_pipeline(
    vcf_path: Path | str,
    output_dir: Path | str | None,
    image: str,
    timeout: int,
) -> tuple[Path, str, subprocess.CompletedProcess[str]]:
    """Run the PharmCAT Docker pipeline; return (out_path, basename, result).

    Does not inspect the produced reports — the caller is responsible
    for discovering and returning report paths (single vs per-sample).
    """
    vcf_path = Path(vcf_path).resolve()
    if not vcf_path.exists():
        raise FileNotFoundError(f"VCF not found: {vcf_path}")

    if not docker_available():
        raise DockerUnavailable(
            "docker is not installed or its daemon is not running. "
            "Install Docker Desktop (or your platform's docker), start "
            "it, and verify `docker info` succeeds."
        )

    out_path = (
        Path(output_dir).resolve() if output_dir else vcf_path.parent
    )
    out_path.mkdir(parents=True, exist_ok=True)

    basename = _vcf_basename(vcf_path)

    # PharmCAT's preprocessor writes a bgzipped copy of the input
    # alongside the input file. If the input directory is read-only
    # (true in CI sandboxes and when input lives in a non-writable
    # location), the preprocessor fails. Copy the VCF into the output
    # directory and mount only that — single read-write mount.
    staged_vcf = out_path / vcf_path.name
    if staged_vcf.resolve() != vcf_path.resolve():
        shutil.copy2(vcf_path, staged_vcf)

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{out_path}:/data",
        image,
        "pharmcat_pipeline",
        "-reporterJson",  # default Reporter output is HTML; opt into JSON
        "-o",
        "/data",
        f"/data/{vcf_path.name}",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PharmCATRunError(
            f"PharmCAT timed out after {timeout}s on {vcf_path.name}. "
            f"Use a longer timeout for full WGS VCFs."
        ) from exc

    if result.returncode != 0:
        raise PharmCATRunError(
            f"PharmCAT exited {result.returncode} on {vcf_path.name}.\n"
            f"stdout (tail):\n{_tail(result.stdout, 30)}\n"
            f"stderr (tail):\n{_tail(result.stderr, 30)}"
        )

    return out_path, basename, result


def _discover_sample_reports(out_path: Path, basename: str) -> tuple[Path, ...]:
    """Return per-sample ``<basename>.Sample_*.report.json`` files, sorted."""
    return tuple(sorted(out_path.glob(f"{basename}.Sample_*.report.json")))


def run_pharmcat(
    vcf_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    image: str = DEFAULT_IMAGE,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> Path:
    """Run the PharmCAT pipeline on a single-sample VCF.

    For multi-sample VCFs (PharmCAT's reporter splits output into
    per-sample files), use :func:`run_pharmcat_multi` instead.

    Args:
        vcf_path: Path to the input VCF (or .vcf.gz / .vcf.bgz).
        output_dir: Where PharmCAT writes its outputs. Defaults to the
            same directory as the input VCF.
        image: Docker image tag for PharmCAT.
        timeout: Subprocess timeout in seconds.

    Returns:
        Path to ``<basename>.report.json``.

    Raises:
        DockerUnavailable: docker isn't installed or daemon not running.
        FileNotFoundError: the VCF doesn't exist.
        PharmCATRunError: PharmCAT exited non-zero, timed out, produced
            no report, or produced per-sample reports (multi-sample VCF
            — use :func:`run_pharmcat_multi`).
    """
    out_path, basename, result = _invoke_pharmcat_pipeline(
        vcf_path, output_dir, image, timeout
    )

    expected_report = out_path / f"{basename}.report.json"
    if expected_report.exists():
        return expected_report

    per_sample = _discover_sample_reports(out_path, basename)
    if per_sample:
        raise PharmCATRunError(
            f"multi-sample input detected: PharmCAT produced "
            f"{len(per_sample)} per-sample report(s) for {basename}. "
            f"Use run_pharmcat_multi() to get all reports."
        )

    raise PharmCATRunError(
        f"PharmCAT did not produce {expected_report}. "
        f"Inspect {out_path} for what was produced.\n"
        f"stdout (tail):\n{_tail(result.stdout, 30)}"
    )


def run_pharmcat_multi(
    vcf_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    image: str = DEFAULT_IMAGE,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> tuple[Path, ...]:
    """Run the PharmCAT pipeline and return one report JSON path per sample.

    PharmCAT's reporter emits ``<basename>.report.json`` for a single-
    sample VCF and ``<basename>.Sample_X.report.json`` per sample for a
    multi-sample VCF. This function returns whichever set was produced:
    a 1-tuple for single-sample input, an N-tuple for an N-sample VCF.

    Returns:
        Tuple of report JSON paths, sorted by filename. Always at least
        one entry on success.

    Raises:
        DockerUnavailable, FileNotFoundError, PharmCATRunError — see
        :func:`run_pharmcat`.
    """
    out_path, basename, result = _invoke_pharmcat_pipeline(
        vcf_path, output_dir, image, timeout
    )

    single = out_path / f"{basename}.report.json"
    if single.exists():
        return (single,)

    per_sample = _discover_sample_reports(out_path, basename)
    if per_sample:
        return per_sample

    raise PharmCATRunError(
        f"PharmCAT did not produce any report.json for {basename} "
        f"under {out_path}. Inspect {out_path} for what was produced.\n"
        f"stdout (tail):\n{_tail(result.stdout, 30)}"
    )


def vcf_to_bundle(
    vcf_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC,
    image: str = DEFAULT_IMAGE,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> Bundle[PGxFinding]:
    """One-shot: run PharmCAT on a single-sample VCF and parse the Bundle.

    For multi-sample VCFs, use :func:`vcf_to_bundles`.
    """
    report = run_pharmcat(
        vcf_path,
        output_dir=output_dir,
        image=image,
        timeout=timeout,
    )
    return parse_pharmcat_json(report, privacy_tier=privacy_tier)


def vcf_to_bundles(
    vcf_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC,
    image: str = DEFAULT_IMAGE,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> tuple[Bundle[PGxFinding], ...]:
    """One-shot: run PharmCAT on a VCF and parse one Bundle per sample.

    Returns a 1-tuple for single-sample input and an N-tuple for an
    N-sample multi-sample VCF.
    """
    reports = run_pharmcat_multi(
        vcf_path,
        output_dir=output_dir,
        image=image,
        timeout=timeout,
    )
    return tuple(
        parse_pharmcat_json(report, privacy_tier=privacy_tier)
        for report in reports
    )


def _tail(text: str, n: int) -> str:
    """Return the last n lines of text — for error context."""
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if lines else "(empty)"
