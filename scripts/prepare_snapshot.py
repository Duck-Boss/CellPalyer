"""Copy a compact, credential-free handoff from an existing experiment workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_project.resolve()
    harness = source / "lexa_text_harness"
    target = REPO / "lexa_text_harness"
    assert source != REPO and harness.is_dir()
    for p in harness.iterdir():
        if p.is_file() and p.suffix in (".py", ".txt", ".json"):
            copy_file(p, target / p.name)
    for p in (source / "LEXA-5913").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts:
            copy_file(p, REPO / p.relative_to(source))
    # These upstream implementations are referenced by the baseline adapters.
    codex = Path("literature/tdc_leaderboard/CODEX-master")
    for p in (harness / codex).rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and (p.suffix in (".py", ".txt", ".md") or p.name in ("LICENSE", "DOCKERFILE")):
            copy_file(p, target / p.relative_to(harness))
    bench = Path("benchmarks/pilot_matched_baselines")
    for p in (harness / bench).rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(harness)
        parts = rel.parts
        if "__pycache__" in parts or any("smoke" in x for x in parts) or p.suffix in (".pt", ".log", ".zip"):
            continue
        if "vendor" in parts:
            if "CPA-main" in parts:
                sub = p.relative_to(harness / bench / "vendor/CPA-main")
                keep = (sub.parts[0] == "cpa" and p.suffix == ".py") or len(sub.parts) == 1 and p.name in ("LICENSE", "README.md", "requirements.txt")
            else:
                keep = p.suffix == ".json"
            if not keep:
                continue
        copy_file(p, target / rel)
    # Preserve finished experiment evidence, not raw API request/response logs.
    runs = harness / "runs/pilot_seen_dose"
    folders = ["prepared", "lexa", "codec_changes", "codec_capacity_oracle",
               "experiment_v2_feedback", "experiment_v3_hierarchy_fixed",
               "experiment_v4_pure_mechanism_fixed"]
    for folder in folders:
        for p in (runs / folder).rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(harness)
            sub = p.relative_to(runs / folder)
            if any(x in sub.parts for x in ("requests", "attempts", "__pycache__")) or p.suffix in (".pt", ".log") or p.name.startswith("."):
                continue
            if folder == "codec_changes" and p.name == "case_state_occupancy.csv":
                continue
            if p.suffix == ".npz":
                if folder == "lexa":
                    keep = p.name in ("exact_code_effects.npz", "pilot_scaffold.npz", "train_codes.npz")
                elif folder in ("prepared", "codec_changes", "codec_capacity_oracle"):
                    keep = False
                elif folder == "experiment_v3_hierarchy_fixed" and "inputs" in sub.parts:
                    keep = False
                else:
                    keep = p.stat().st_size < 5_000_000
                if not keep:
                    continue
            if p.suffix not in (".py", ".txt", ".json", ".jsonl", ".csv", ".png", ".html", ".svg", ".npz"):
                continue
            if len(sub.parts) > 1 and folder == "experiment_v2_feedback":
                continue
            copy_file(p, target / rel)
    for p in (harness / "literature").rglob("*"):
        if not p.is_file() or "CODEX-master" in p.parts or "cpa-reproducibility-main" in p.parts:
            continue
        if p.suffix in (".json", ".txt") and p.stat().st_size < 5_000_000:
            copy_file(p, target / p.relative_to(harness))
    # The official-leaderboard audit remains research evidence, not a result.
    tdc = harness / "benchmarks/tdc_sciplex2"
    for p in tdc.rglob("*"):
        if not p.is_file() or any(x in p.relative_to(tdc).parts for x in ("vendor", "range_parts", "__pycache__")):
            continue
        if p.suffix in (".py", ".json", ".txt") and p.stat().st_size < 2_000_000:
            copy_file(p, target / p.relative_to(harness))
    original = harness / bench
    result = REPO / "results"
    result.mkdir(exist_ok=True)
    for name in ("scores_by_well.csv", "scores_by_drug.csv", "scores_by_seed.csv", "scores_summary.csv",
                 "comparison.png", "training_curves.png", "RESULTS.html", "RESULTS.txt", "TEACHER_BRIEF.txt",
                 "teacher_workflow.svg", "teacher_workflow.mmd", "teacher_comparison.csv",
                 "protocol.json", "score_manifest.json", "independent_score_audit.json", "completion_manifest.json"):
        copy_file(original / name, result / name)
    protocol = json.loads((original / "protocol.json").read_text())
    for name, info in protocol["inputs"].items():
        assert sha(Path(info["path"])) == info["sha256"], name
    assert sha(runs / "prepared/evaluation.npz") == protocol["evaluation_archive_sha256"]
    with np.load(runs / "prepared/evaluation.npz", allow_pickle=False) as z:
        evaluation = {k: z[k].copy() for k in z.files}
    cases = json.loads((original / "data/prediction_cases.json").read_text())["cases"]
    full = {c["case_id"]: c for c in json.loads((runs / "prepared/cases.json").read_text())["cases"]}
    ids = np.array([c["case_id"] for c in cases])
    genes = evaluation["genes"].astype(str)
    treated, control = [], []
    for c in cases:
        tx = evaluation["x_test"][evaluation["test_case_ids"].astype(str) == c["case_id"]].astype(float)
        cx = evaluation["control_x"][np.isin(evaluation["control_cell_ids"].astype(str), c["control_cell_ids"])].astype(float)
        assert len(tx) == full[c["case_id"]]["n_treated"] and len(cx) == full[c["case_id"]]["n_control"]
        treated.append(tx.mean(0));control.append(cx.mean(0))
    treated, control = np.array(treated), np.array(control)
    model_names, seeds, views, deltas = [], [], [], []
    source_hashes = {"evaluation.npz": sha(runs / "prepared/evaluation.npz")}
    for model in ("cpa", "codex"):
        for seed in (0, 1, 2):
            p = original / "models" / model / f"seed_{seed}" / "predictions.npz"
            source_hashes[str(p.relative_to(harness))] = sha(p)
            with np.load(p, allow_pickle=False) as z:
                assert np.array_equal(z["case_ids"].astype(str), ids) and np.array_equal(z["genes"].astype(str), genes)
                for view, delta in [("native", z["pred_treated_mean"].astype(float) - control),
                                    ("calibrated", z["pred_treated_mean"].astype(float) - z["pred_control_mean"].astype(float))]:
                    model_names.append(model);seeds.append(seed);views.append(view);deltas.append(delta)
    p = original / "references/predictions.npz"
    source_hashes[str(p.relative_to(harness))] = sha(p)
    with np.load(p, allow_pickle=False) as z:
        assert np.array_equal(z["case_ids"].astype(str), ids) and np.array_equal(z["genes"].astype(str), genes)
        for i, name in enumerate(z["model_names"]):
            model_names.append(str(name));seeds.append(-1);views.append("existing");deltas.append(z["delta"][i].astype(float))
    replay = result / "replay.npz"
    np.savez_compressed(replay, case_ids=ids, genes=genes,
        drugs=np.array([c["drug"] for c in cases]),
        n_treated=np.array([full[c["case_id"]]["n_treated"] for c in cases]),
        n_control=np.array([full[c["case_id"]]["n_control"] for c in cases]),
        treated_mean=treated, control_mean=control,
        model_names=np.array(model_names), seeds=np.array(seeds), views=np.array(views), predicted_delta=np.array(deltas))
    omitted = []
    for p in harness.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(harness)
        if (p.suffix == ".pt" or str(rel) in ("runs/pilot_seen_dose/prepared/prepared.npz", "runs/pilot_seen_dose/prepared/evaluation.npz")) and not (target / rel).exists():
            omitted.append({"relative_path": str(rel), "local_path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha(p)})
    source_meta = json.loads((runs / "lexa/metadata.json").read_text())
    for name, digest in source_meta["source_sha256"].items():
        assert sha(REPO / "LEXA-5913/lexa" / name) == digest, "LEXA author source changed"
    manifest = {
        "exported_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "source_harness": str(harness), "remote_target": "Duck-Boss/CellPalyer",
        "scope": "existing completed experiments; no new fitting or API calls",
        "replay_sha256": sha(replay), "source_prediction_hashes": source_hashes,
        "original_score_manifest_sha256": sha(original / "score_manifest.json"),
        "omitted_local_artifacts": omitted,
        "omissions": ["raw public datasets", "model checkpoints", "raw API request/response traces", "large third-party archives/notebooks", "credentials"],
        "historical_paths": "Original JSON records preserve their original absolute paths; use scripts/verify_results.py for portable result verification.",
    }
    (result / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    original_wrapper = harness / bench / "run_cpa.py"
    snapshot = REPO / "source_snapshots/run_cpa_original.py"
    copy_file(original_wrapper, snapshot)
    portable = target / bench / "run_cpa.py"
    code = original_wrapper.read_text()
    anchor = '    ap.add_argument("--seed", type=int, required=True)'
    assert code.count(anchor) == 1
    code = code.replace(anchor, anchor + '\n    ap.add_argument("--prepared", type=Path, default=ROOT / "runs/pilot_seen_dose/prepared")')
    anchor = '    prepared = ROOT / "runs/pilot_seen_dose/prepared"'
    assert code.count(anchor) == 1
    portable.write_text(code.replace(anchor, '    prepared = args.prepared.resolve()'))
    files = [p for p in REPO.rglob("*") if p.is_file() and ".git" not in p.parts]
    print(json.dumps({"files": len(files), "bytes": sum(p.stat().st_size for p in files),
                      "replay_bytes": replay.stat().st_size, "prediction_variants": len(deltas), "cases": len(ids)}))


if __name__ == "__main__":
    main()
