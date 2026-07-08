"""Run garak as a subprocess and stream its progress + parsed results.

Flow:
  1. Write a garak REST-generator config pointing at settings.garak_target_url
     (the app's own /garak/generate endpoint).
  2. Spawn `python -m garak --model_type rest ...` for the chosen probes.
  3. Stream stdout line-by-line to the caller (the UI shows a live log).
  4. When it finishes, parse the .report.jsonl garak wrote and return a
     per-check pass/fail summary for visualisation.

"Pass" in garak means the model RESISTED the attack (no vulnerability hit), so
a HIGH pass-rate = SAFER. We surface it that way in the UI.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from config import ROOT, settings

# Curated probe catalogue for the UI. Keys are garak probe specs (module or
# module.Class); descriptions explain the attack. "rag" flags the ones most
# relevant to a retrieval pipeline (injection hidden in retrieved text, leakage).
PROBE_CATALOG: list[dict] = [
    {"id": "promptinject", "label": "Prompt injection",
     "desc": "Hijack the model with instructions embedded in the input.", "rag": True},
    {"id": "latentinjection", "label": "Latent (indirect) injection",
     "desc": "Malicious instructions hidden in documents/context — the classic RAG attack.", "rag": True},
    {"id": "leakreplay", "label": "Training-data / context leakage",
     "desc": "Coax the model into reciting memorised or provided text verbatim.", "rag": True},
    {"id": "dan.Dan_11_0", "label": "Jailbreak (DAN 11.0)",
     "desc": "Classic 'Do Anything Now' role-play jailbreak.", "rag": False},
    {"id": "encoding", "label": "Encoded-payload injection",
     "desc": "Smuggle instructions past filters via base64/rot13/etc.", "rag": True},
    {"id": "xss", "label": "Markdown / HTML / XSS exfiltration",
     "desc": "Trick the model into emitting active markup that leaks data.", "rag": True},
    {"id": "malwaregen", "label": "Malware generation",
     "desc": "Ask for working malicious code.", "rag": False},
    {"id": "packagehallucination", "label": "Package hallucination",
     "desc": "Model invents non-existent (squattable) software packages.", "rag": False},
    {"id": "goodside", "label": "Adversarial trickery (goodside)",
     "desc": "Known prompt tricks that reliably confuse LLMs.", "rag": False},
    {"id": "glitch", "label": "Glitch tokens",
     "desc": "Tokens that destabilise the model's output.", "rag": False},
]

DEFAULT_PROBES = ["promptinject"]

# Strip ANSI colour codes from garak's stdout before showing them in the UI.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def garak_available() -> bool:
    """True if the garak package is importable (i.e. installed)."""
    return importlib.util.find_spec("garak") is not None


def _garak_runs_dir() -> Path | None:
    """Where garak writes reports (xdg_data_home/garak/garak_runs)."""
    try:
        from xdg_base_dirs import xdg_data_home

        return xdg_data_home() / "garak" / "garak_runs"
    except Exception:  # noqa: BLE001
        return None


def _write_rest_config(target_url: str, path: Path) -> None:
    """garak REST-generator options: POST {"prompt": "<attack>"} and read the
    answer back from the JSON "output" field."""
    cfg = {
        "rest": {
            "RestGenerator": {
                "name": "gitagpt-target",
                "uri": target_url,
                "method": "post",
                "headers": {"Content-Type": "application/json"},
                "req_template_json_object": {"prompt": "$INPUT"},
                "response_json": True,
                "response_json_field": "output",
                "request_timeout": settings.garak_request_timeout,
            }
        }
    }
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def run_scan(probes: list[str] | None, generations: int, label: str):
    """Generator yielding SSE-shaped dicts: log / status / result / error."""
    if not garak_available():
        yield {"type": "error",
               "message": "garak is not installed. Install it with:  pip install garak"}
        return

    settings.garak_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    # garak resolves --report_prefix inside its OWN run dir and won't create
    # sub-paths, so this must be a bare name (not a path). We locate the written
    # report afterwards by scraping stdout / globbing garak's run dir.
    prefix = f"gitagpt_{ts}"
    cfg_path = settings.garak_dir / f"config_{ts}.json"
    _write_rest_config(settings.garak_target_url, cfg_path)

    probe_list = probes or DEFAULT_PROBES
    probe_arg = ",".join(probe_list)
    gens = max(1, int(generations or 1))

    cmd = [
        sys.executable, "-m", "garak",
        "--target_type", "rest",
        "-G", str(cfg_path),
        "--probes", probe_arg,
        "--generations", str(gens),
        "--report_prefix", prefix,
    ]

    yield {"type": "status", "message": f"Scanning “{label}” with probes: {probe_arg}"}
    yield {"type": "log", "line": "$ " + " ".join(cmd)}

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, cwd=str(ROOT),
            # garak prints emoji + ANSI colour; on Windows the default cp1252
            # decode raises UnicodeDecodeError, so force UTF-8 and never crash.
            encoding="utf-8", errors="replace",
        )
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": f"Failed to launch garak: {exc}"}
        return

    report_path: str | None = None
    for line in proc.stdout:  # type: ignore[union-attr]
        line = _ANSI_RE.sub("", line).rstrip("\n")
        if not line.strip():
            continue
        yield {"type": "log", "line": line}
        m = re.search(r"(\S+\.report\.jsonl)", line)
        if m:
            report_path = m.group(1)
    proc.wait()

    # Fall back to newest report in garak's own run dir if we couldn't scrape it.
    if not report_path or not Path(report_path).exists():
        runs_dir = _garak_runs_dir()
        if runs_dir:
            candidates = sorted(runs_dir.glob(f"{prefix}*.report.jsonl")) \
                or sorted(runs_dir.glob("gitagpt_*.report.jsonl"))
            report_path = str(candidates[-1]) if candidates else None

    if not report_path or not Path(report_path).exists():
        yield {"type": "error",
               "message": "Scan finished but no report file was produced — see the log above."}
        return

    rows, summary = _parse_report(report_path)
    summary["label"] = label
    summary["return_code"] = proc.returncode
    yield {"type": "result", "summary": summary, "probes": rows, "report": report_path}


def _parse_report(path: str) -> tuple[list[dict], dict]:
    """Aggregate garak 'eval' records into per-check pass/fail rows."""
    checks: dict[str, dict] = {}
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if rec.get("entry_type") != "eval":
            continue
        probe = rec.get("probe", "?")
        detector = rec.get("detector", "")
        passed = int(rec.get("passed") or 0)
        # garak 0.15 uses total_evaluated/fails; older builds used "total".
        fails = rec.get("fails")
        total = rec.get("total_evaluated") or rec.get("total")
        if total is None:
            total = passed + int(fails or 0)
        total = int(total)
        key = f"{probe} / {detector}" if detector else probe
        slot = checks.setdefault(key, {"probe": probe, "detector": detector,
                                        "passed": 0, "total": 0})
        slot["passed"] += passed
        slot["total"] += total

    rows: list[dict] = []
    tot_p = tot_t = 0
    for name, v in sorted(checks.items()):
        total, passed = v["total"], v["passed"]
        rate = (passed / total * 100) if total else 0.0
        rows.append({
            "name": name,
            "probe": v["probe"],
            "detector": v["detector"],
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "pass_rate": round(rate, 1),
        })
        tot_p += passed
        tot_t += total

    # Most vulnerable (lowest pass-rate) first — that's what the user cares about.
    rows.sort(key=lambda r: r["pass_rate"])
    overall = round(tot_p / tot_t * 100, 1) if tot_t else 0.0
    summary = {
        "overall_pass_rate": overall,
        "total_passed": tot_p,
        "total_failed": tot_t - tot_p,
        "total": tot_t,
        "num_checks": len(rows),
    }
    return rows, summary
