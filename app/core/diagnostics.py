"""Startup readiness check.

Runs every time the app starts, prints a readable report, and never blocks
startup on an optional provider being absent. The point is that a person
looking at the logs immediately sees what works, what is missing, and
whether that matters.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..ai.providers import PROVIDERS, Provider


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fatal: bool = False

    def line(self) -> str:
        mark = "PASS" if self.ok else ("FAIL" if self.fatal else "WARN")
        return f"  [{mark}] {self.name}: {self.detail}"


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def fatal_failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.fatal]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.fatal]

    def render(self) -> str:
        lines = ["", "=" * 62, " FoundIt — startup readiness", "=" * 62]
        lines.extend(c.line() for c in self.checks)
        lines.append("-" * 62)
        if self.fatal_failures:
            lines.append(f" {len(self.fatal_failures)} FATAL problem(s) — demo cannot start")
            for check in self.fatal_failures:
                lines.append(f"   -> {check.detail}")
        elif self.warnings:
            lines.append(
                f" Ready. {len(self.warnings)} optional feature(s) unavailable; "
                "answers stay honest but thinner."
            )
        else:
            lines.append(" Ready. Every configured component responded.")
        lines.append("=" * 62)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "ok": not self.fatal_failures,
            "checks": [
                {"name": c.name, "ok": c.ok, "fatal": c.fatal, "detail": c.detail}
                for c in self.checks
            ],
            "warnings": len(self.warnings),
        }


ROOT = Path(__file__).resolve().parent.parent.parent


def check_python(report: Report) -> None:
    import sys

    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    ok = sys.version_info >= (3, 11)
    report.add(Check("python", ok, f"{version}", fatal=True))


def check_ffmpeg(report: Report) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        report.add(Check("ffmpeg", True, "ffmpeg and ffprobe found"))
    else:
        report.add(Check(
            "ffmpeg", False,
            "ffmpeg/ffprobe missing — video ingest disabled, demo mode still works",
            fatal=False,
        ))


def check_media_decoding(report: Report) -> None:
    """Prove decoding works, not just that the binary exists."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        report.add(Check("media_decode", False, "skipped: no ffmpeg", fatal=False))
        return
    import subprocess

    sample = ROOT / "fixtures" / "media" / "synthetic_test.mp4"
    if not sample.exists():
        report.add(Check(
            "media_decode", True,
            "no sample clip present; ffmpeg available",
            fatal=False,
        ))
        return
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(sample),
         "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60,
    )
    report.add(Check(
        "media_decode",
        result.returncode == 0,
        "sample clip decodes" if result.returncode == 0 else f"decode failed: {result.stderr[:120]}",
        fatal=False,
    ))


def check_schemas(report: Report) -> None:
    schema_dir = ROOT / "schemas"
    expected = [
        "video_source", "observation", "item_event", "report_window",
        "evidence", "search_result", "handover_record",
    ]
    missing = [s for s in expected if not (schema_dir / f"{s}.schema.json").exists()]
    if missing:
        report.add(Check("schemas", False, f"missing: {', '.join(missing)}", fatal=True))
        return
    for name in expected:
        try:
            json.loads((schema_dir / f"{name}.schema.json").read_text())
        except json.JSONDecodeError as error:
            report.add(Check("schemas", False, f"{name} is invalid JSON: {error}", fatal=True))
            return
    report.add(Check("schemas", True, f"{len(expected)} contracts load"))


def check_storage(report: Report) -> None:
    path = os.environ.get("DATABASE_PATH", str(ROOT / "data" / "foundit.db"))
    try:
        store_root = Path(path)
        store_root.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(store_root))
        connection.execute("CREATE TABLE IF NOT EXISTS _probe (x INTEGER)")
        connection.execute("DROP TABLE _probe")
        connection.commit()
        connection.close()
        report.add(Check("storage", True, f"sqlite writable at {path}"))
    except Exception as error:  # noqa: BLE001 - surface any storage failure
        report.add(Check("storage", False, f"unusable at {path}: {error}", fatal=True))


def check_fixtures(report: Report) -> None:
    fixture_dir = ROOT / "fixtures"
    required = [
        "video_sources.json", "observations.json", "item_events.json",
        "report_windows.json", "evidence.json", "search_results.json",
        "handover_records.json",
    ]
    missing = [f for f in required if not (fixture_dir / f).exists()]
    if missing:
        report.add(Check("fixtures", False, f"missing: {', '.join(missing)}", fatal=True))
        return
    try:
        for name in required:
            json.loads((fixture_dir / name).read_text())
        report.add(Check("fixtures", True, f"{len(required)} fixture files load"))
    except json.JSONDecodeError as error:
        report.add(Check("fixtures", False, f"invalid JSON: {error}", fatal=True))


def check_providers(report: Report) -> None:
    for provider in PROVIDERS:
        status = provider.status()
        if status == "absent_required":
            report.add(Check(
                f"ai:{provider.name}", False,
                f"{provider.env_var} not set — {provider.without_it}",
                fatal=False,
            ))
        elif status == "absent_optional":
            report.add(Check(
                f"ai:{provider.name}", True,
                f"not configured — {provider.without_it[:90]}",
                fatal=False,
            ))
        elif status == "built_in":
            report.add(Check(f"ai:{provider.name}", True, provider.purpose[:70]))
        else:
            report.add(Check(f"ai:{provider.name}", True, f"configured via {provider.env_var}"))


def check_jinja(report: Report) -> None:
    ok = importlib.util.find_spec("jinja2") is not None
    report.add(Check(
        "templates", ok,
        "jinja2 available" if ok else "jinja2 missing — API works, web UI disabled",
        fatal=False,
    ))


def run_checks() -> Report:
    report = Report()
    check_python(report)
    check_ffmpeg(report)
    check_media_decoding(report)
    check_schemas(report)
    check_storage(report)
    check_fixtures(report)
    check_jinja(report)
    check_providers(report)
    return report


def provider_summary() -> list[dict]:
    return [
        {
            "name": p.name,
            "capability": p.capability.value,
            "requirement": p.requirement.value,
            "env_var": p.env_var,
            "status": p.status(),
            "purpose": p.purpose,
            "without_it": p.without_it,
        }
        for p in PROVIDERS
    ]
