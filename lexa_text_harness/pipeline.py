"""Reproducible sciPlex2 -> frozen LEXA -> text-agent experiment pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def command(script, *arguments):
    invocation = [sys.executable, str(ROOT / script), *map(str, arguments)]
    print("STAGE", script, flush=True)
    subprocess.run(invocation, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "offline", "download", "prepare", "fit", "annotate", "benchmark", "score"], default="all")
    parser.add_argument("--work", type=Path, default=ROOT / "runs/pilot_seen_dose")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/sciplex2")
    parser.add_argument("--protocol", choices=["seen_drug_unseen_dose", "leave_drug_out"], default="seen_drug_unseen_dose")
    parser.add_argument("--holdout-drug", choices=["BMS", "Dex", "Nutlin", "SAHA"])
    parser.add_argument("--device", default="cpu", help="CPU by default; use cuda:N explicitly for an allocated GPU")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--case-limit", type=int, default=4, help="0 means every ready held-out well")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-file")
    parser.add_argument("--methods", default="one_shot,agent")
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--experiment-name", help="Result subdirectory; otherwise use active_experiment.json or experiment")
    args = parser.parse_args()
    args.work = args.work.resolve()
    args.raw = args.raw.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / ".pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This work directory already has a running pipeline")
        return run(args)


def run(args):
    active = args.work / 'active_experiment.json'
    experiment_name = args.experiment_name or (json.loads(active.read_text())['name'] if active.exists() else 'experiment')
    if Path(experiment_name).name != experiment_name or experiment_name in {'.', '..'}:
        raise ValueError('experiment-name must be a single directory name')
    prepared, fitted, experiment = args.work / "prepared", args.work / "lexa", args.work / experiment_name
    stages = ["download", "prepare", "fit", "annotate", "benchmark", "score"]
    selected = stages if args.stage == "all" else stages[:3] if args.stage == "offline" else [args.stage]
    def status(state, stage):
        (args.work / "pipeline_status.json").write_text(json.dumps({
            "state": state, "stage": stage, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": args.protocol, "work": str(args.work), "model": args.model,
            "experiment": str(experiment),
        }, indent=2) + "\n")
    for stage in selected:
        status("running", stage)
        try:
            if stage == "download":
                command("download_data.py", "--output-dir", args.raw)
            elif stage == "prepare":
                extra = ["--holdout-drug", args.holdout_drug] if args.holdout_drug else []
                if not (prepared / "COMPLETE.json").exists():
                    command("prepare_data.py", "--raw", args.raw, "--output", prepared,
                            "--protocol", args.protocol, *extra)
                manifest = json.loads((prepared / "manifest.json").read_text())
                if manifest.get("protocol") != args.protocol:
                    raise ValueError("Prepared protocol differs; use a new work directory")
                if manifest.get("holdout_drug") != args.holdout_drug:
                    raise ValueError("Prepared holdout drug differs; use a new work directory")
            elif stage == "fit":
                from tools import sha256
                complete = fitted / "COMPLETE.json"
                if complete.exists():
                    value = json.loads(complete.read_text())
                    if value.get("smoke_only") or value.get("prepared_sha256") != sha256(prepared / "prepared.npz"):
                        raise ValueError("Existing fit is incomplete/smoke-only or belongs to another prepared dataset")
                    metadata = json.loads((fitted / 'metadata.json').read_text())
                    if metadata['training']['epochs_requested'] != args.epochs:
                        raise ValueError("Requested training horizon differs from the completed fit; use a new work directory")
                else:
                    extra = ["--resume"] if (fitted / "last.pt").exists() else []
                    command("fit_lexa.py", "--prepared", prepared / "prepared.npz", "--output", fitted,
                            "--device", args.device, "--epochs", args.epochs, "--threads", args.threads, *extra)
                from tools import Workspace
                request = fitted / "card_export_request.json"
                request.write_text(json.dumps({"mode": "functional_edit", "text": "Prepare signed evidence cards; no biological editing requested."}))
                workspace = Workspace(str(fitted / "exact_code_effects.npz"), str(fitted / "baseline.npz"),
                                      str(request), "mse_log1p", str(fitted / "card_export"))
                workspace.export_cards(fitted / "evidence_cards.jsonl")
            elif stage in {"annotate", "benchmark"}:
                available_key = bool(os.environ.get("DEEPSEEK_API_KEY")) or bool(args.api_key_file and Path(args.api_key_file).is_file())
                if not available_key:
                    status("waiting_for_deepseek_api_key", stage)
                    print("Offline artifacts are ready. Set DEEPSEEK_API_KEY and rerun the same command.", flush=True)
                    return 0
                key_args = ["--api-key-file", args.api_key_file] if args.api_key_file else []
                common = ["--model", args.model, "--base-url", args.base_url, *key_args]
                if stage == "annotate":
                    extra = ["--resume"] if (fitted / "labels.json").exists() else []
                    command("annotate_cards.py", "--effects", fitted / "exact_code_effects.npz",
                            "--output", fitted / "labels.json", *common, *extra)
                else:
                    command("experiment.py", "agents", "--prepared", prepared, "--fitted", fitted,
                            "--output", experiment, "--case-limit", args.case_limit,
                            "--methods", args.methods, "--max-rounds", args.max_rounds, *common)
            elif stage == "score":
                command("experiment.py", "score", "--prepared", prepared, "--fitted", fitted, "--output", experiment)
                command("representation_diagnostic.py", "--prepared", prepared, "--fitted", fitted,
                        "--output", experiment / "representation_diagnostic.json")
        except Exception:
            status("failed", stage)
            raise
        status("completed", stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
