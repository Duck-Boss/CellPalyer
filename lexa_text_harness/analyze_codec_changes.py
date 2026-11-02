#!/usr/bin/env python3
"""Describe observed treatment/control occupancy shifts through a frozen codec.

All available doses are included, including training cells. This is a posthoc
representation diagnostic, not intervention prediction or paired-cell tracking.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import fit_lexa
from representation_diagnostic import delta_metrics, reconstruction_mse


def occupancy(codes, states):
    if codes.ndim != 2 or len(codes) == 0:
        raise ValueError("Expected nonempty cell-by-axis codes")
    result = np.stack([np.bincount(codes[:, a], minlength=states) / len(codes)
                       for a in range(codes.shape[1])])
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-12)
    return result


def divergence(p, q):
    m = (p + q) / 2
    def kl(v):
        active = v > 0
        terms = np.zeros_like(v)
        terms[active] = v[active] * np.log2(v[active] / m[active])
        return terms.sum(axis=-1)
    return .5 * np.abs(q - p).sum(axis=-1), .5 * (kl(p) + kl(q))


def metrics(actual, decoded):
    result = delta_metrics(actual, decoded)
    top = np.argsort(-np.abs(actual), kind="stable")[:100]
    result["top100_rmse"] = float(np.sqrt(np.mean((decoded[top] - actual[top]) ** 2)))
    a = np.where(actual[top] >= .001, 1, np.where(actual[top] <= -.001, -1, 0))
    b = np.where(decoded[top] >= .001, 1, np.where(decoded[top] <= -.001, -1, 0))
    active = a != 0
    result["top100_direction_agreement"] = float((a[active] == b[active]).mean()) if active.any() else None
    result["top100_n_direction_genes"] = int(active.sum())
    return result


def export_tables(output, prefix, items, labels, axes, states):
    state_rows, axis_rows, metric_rows = [], [], []
    for item in items:
        keys = item["keys"]
        p, q, effects = item["pcontrol"], item["ptreated"], item["axis_delta_genes"]
        tv, js = divergence(p, q)
        for a in range(axes):
            axis_row = {**keys, "axis_id": a, "total_variation": float(tv[a]),
                        "js_divergence_base2": float(js[a]),
                        "axis_effect_l2": float(np.linalg.norm(effects[a]))}
            if "mean_well_total_variation" in item:
                axis_row.update({name:float(item[name][a])for name in
                                 ["mean_well_total_variation","mean_well_js_divergence_base2","mean_well_axis_effect_l2"]})
            axis_rows.append(axis_row)
            for s in range(states):
                token = f"A{a}_C{s}"
                state_row = {**keys, "axis_id": a, "state_id": s,
                                   "control_fraction": float(p[a,s]),
                                   "treated_fraction": float(q[a,s]),
                                   "delta_fraction": float(q[a,s] - p[a,s]),
                                   "label": token, "name": labels.get(token, {}).get("name", "")}
                if "delta_fraction_well_sd" in item:
                    state_row["delta_fraction_well_sd"] = float(item["delta_fraction_well_sd"][a,s])
                state_rows.append(state_row)
        metric_row={**keys, **metrics(item["observed_delta"], item["decoded_delta"]),
                    "treated_reconstruction_mse": item["treated_reconstruction_mse"],
                    "control_reconstruction_mse": item["control_reconstruction_mse"]}
        if "mean_well_metrics" in item:
            metric_row.update({"mean_vector_"+name:value for name,value in metrics(item["observed_delta"],item["decoded_delta"]).items()})
            metric_row.update({"mean_well_"+name:value for name,value in item["mean_well_metrics"].items()})
        metric_rows.append(metric_row)
    for suffix, rows in [("state_occupancy", state_rows), ("axis_changes", axis_rows),
                         ("codec_metrics", metric_rows)]:
        pd.DataFrame(rows).to_csv(output / f"{prefix}_{suffix}.csv", index=False)


def run(args):
    work, output = args.work, args.output or args.work / "codec_changes"
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed output exists; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    prepared, fitted = work / "prepared", work / "lexa"
    checkpoint_path = fitted / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if fit_lexa.file_sha256(prepared / "prepared.npz") != checkpoint["prepared_sha256"]:
        raise ValueError("Prepared archive differs from frozen checkpoint")
    metadata = json.loads((fitted / "metadata.json").read_text())
    repo = Path(metadata.get("source_repository", str(fit_lexa.DEFAULT_REPO)))
    for filename, expected in checkpoint.get("source_sha256", {}).items():
        if fit_lexa.file_sha256(repo / "lexa" / filename) != expected:
            raise ValueError(f"Original model source changed: {filename}")
    if checkpoint["config"]["reconstruction_distribution"] != "mse_log1p":
        raise ValueError("Exact additive expression decomposition requires mse_log1p")

    with np.load(prepared / "prepared.npz", allow_pickle=False) as a:
        genes, symbols = a["genes"].astype(str), a["gene_symbols"].astype(str)
        chunks = [(name, a[x].copy(), a[ids].astype(str)) for name, x, ids in
                  [("train", "x_train", "train_cell_ids"), ("val", "x_val", "val_cell_ids"),
                   ("conditioning_control", "control_x", "control_cell_ids")]]
    with np.load(prepared / "evaluation.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(genes, a["genes"].astype(str))
        chunks += [(name, a[x].copy(), a[ids].astype(str)) for name, x, ids in
                   [("test", "x_test", "test_cell_ids"),
                    ("evaluation_control", "control_x", "control_cell_ids")]]
    if genes.tolist() != checkpoint["gene_vocab"]:
        raise ValueError("Checkpoint gene vocabulary/order mismatch")
    vectors, sources, duplicate_count = {}, {}, 0
    input_counts = {}
    for name, values, ids in chunks:
        input_counts[name] = len(ids)
        if len(set(ids)) != len(ids) or values.shape != (len(ids), len(genes)):
            raise ValueError(f"Invalid archive dimensions/duplicate IDs: {name}")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid normalized expression: {name}")
        for cell, row in zip(ids, values):
            if cell in vectors:
                if not np.array_equal(vectors[cell], row):
                    raise ValueError(f"Repeated cell expression differs: {cell}")
                duplicate_count += 1
            else:
                vectors[cell] = row.copy()
                sources[cell] = []
            sources[cell].append(name)
    cell_ids = np.asarray(sorted(vectors))
    observed = np.stack([vectors[c] for c in cell_ids]).astype(np.float32)
    del chunks, vectors
    cell_meta_all = pd.read_csv(prepared / "metadata.csv")
    if cell_meta_all.cell_id.duplicated().any():
        raise ValueError("Repeated cell IDs in metadata")
    cell_meta = cell_meta_all.set_index("cell_id").loc[cell_ids].reset_index()
    expected = set(cell_meta_all.loc[cell_meta_all.split.isin(["train", "val", "test"]), "cell_id"])
    if set(cell_ids) != expected:
        raise ValueError("Deduplicated archive union differs from eligible metadata cells")
    cell_meta["metadata_index"] = np.arange(len(cell_meta))
    cell_meta["archive_sources"] = ["|".join(sources[c]) for c in cell_ids]
    cell_meta["seen_in_training"] = cell_meta.split.eq("train")
    cell_meta["replicate"] = cell_meta["replicate"].astype(int)

    torch.set_num_threads(args.threads)
    model_class, _ = fit_lexa.import_original(repo)
    model = model_class(**checkpoint["config"], module_mask_prior=checkpoint["module_mask_prior"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    codes = np.empty((len(observed), checkpoint["config"]["num_axes"]), dtype=np.int16)
    reconstructed = np.empty_like(observed)
    with torch.inference_mode():
        for i in range(0, len(observed), args.batch_size):
            x = torch.from_numpy(observed[i:i+args.batch_size])
            q, ix, *_ = model.quantize(model.encode(x))
            codes[i:i+len(x)] = ix.numpy().astype(np.int16)
            reconstructed[i:i+len(x)] = model.decode(q).numpy()
    with np.load(fitted / "exact_code_effects.npz", allow_pickle=False) as a:
        np.testing.assert_array_equal(genes, a["genes"].astype(str))
        effects = a["contributions"].astype(np.float64)
    axes, states, n_genes = effects.shape
    if codes.shape != (len(cell_ids), axes) or codes.min()<0 or codes.max()>=states:
        raise ValueError("Invalid inferred code IDs")
    exact_reconstructed = effects[np.arange(axes)[None,:], codes].sum(axis=1)
    cell_error = float(np.max(np.abs(exact_reconstructed - reconstructed)))
    np.testing.assert_allclose(exact_reconstructed, reconstructed, rtol=1e-5, atol=2e-6)
    cell_meta.to_csv(output / "cell_metadata.csv", index=False)
    np.savez_compressed(output / "cell_codes.npz", cell_ids=cell_ids, genes=genes,
                        gene_symbols=symbols, codes=codes, reconstructed_x=reconstructed,
                        observed_x=observed, metadata_index=np.arange(len(cell_ids)))
    labels_payload = json.loads((fitted / "labels.json").read_text())
    if labels_payload["effects_sha256"] != fit_lexa.file_sha256(fitted / "exact_code_effects.npz"):
        raise ValueError("Annotations refer to different exact effects")
    labels = labels_payload["labels"]
    grouped = cell_meta.groupby(["drug", "dose_uM", "plate", "replicate"], sort=True).indices
    trained_conditions = {(str(r.drug),float(r.dose_uM)) for r in cell_meta.itertuples()
                          if r.split == "train" and r.dose_uM>0}
    cases, excluded, invariant_errors = [], [], []
    annotated_treatments=cell_meta_all.loc[cell_meta_all.parse_status.eq("ok") & cell_meta_all.dose_uM.gt(0)]
    all_treatment_groups=annotated_treatments.groupby(["drug","dose_uM","plate","replicate"],sort=True).indices
    for drug, dose, plate, rep in all_treatment_groups:
        rep=int(rep)
        ti=grouped.get((drug,dose,plate,rep),np.asarray([],dtype=int))
        ci = grouped.get((drug, 0., plate, rep), np.asarray([], dtype=int))
        cid = f"drug={drug}|dose_uM={dose:g}"
        split_values = sorted(cell_meta.iloc[ti].split.unique())
        if len(split_values) > 1:
            raise ValueError("A treatment well spans split boundaries")
        split = split_values[0] if split_values else "excluded_qc_no_cells"
        keys = {"case_id": f"{cid}|plate={plate}|replicate={rep}", "condition_id": cid,
                "drug": drug, "dose_uM": float(dose), "plate": plate, "replicate": int(rep),
                "split": split, "case_split":split, "seen_in_training": split == "train",
                "condition_seen_in_training": (drug,float(dose)) in trained_conditions,
                "n_control": len(ci), "n_treated": len(ti)}
        if min(len(ci), len(ti)) < args.min_cells:
            excluded.append({**keys,"status":"small_sample", "reason":f"Each matched group requires >= {args.min_cells} cells"})
            continue
        pc, pt = occupancy(codes[ci], states), occupancy(codes[ti], states)
        np.testing.assert_allclose((pt-pc).sum(axis=1), 0, atol=1e-12)
        axis_delta = np.einsum("as,asg->ag", pt-pc, effects)
        observed_delta = observed[ti].mean(axis=0,dtype=np.float64)-observed[ci].mean(axis=0,dtype=np.float64)
        decoded_delta = reconstructed[ti].mean(axis=0,dtype=np.float64)-reconstructed[ci].mean(axis=0,dtype=np.float64)
        invariant_errors.append(float(np.max(np.abs(axis_delta.sum(axis=0)-decoded_delta))))
        np.testing.assert_allclose(axis_delta.sum(axis=0), decoded_delta, rtol=1e-5, atol=2e-6)
        cases.append({"keys":keys,"pcontrol":pc,"ptreated":pt,"axis_delta_genes":axis_delta,
                      "observed_delta":observed_delta,"decoded_delta":decoded_delta,
                      "treated_reconstruction_mse":reconstruction_mse(observed[ti],reconstructed[ti]),
                      "control_reconstruction_mse":reconstruction_mse(observed[ci],reconstructed[ci])})
    if not cases:
        raise ValueError("No ready matched cases")
    pd.DataFrame([dict(c["keys"],status="ready") for c in cases]).to_csv(output/"ready_cases.csv",index=False)
    pd.DataFrame(excluded).to_csv(output/"low_sample_exclusions.csv",index=False)
    eligibility=[]
    for drug,dose in sorted({(d,float(v))for d,v,_,_ in all_treatment_groups}):
        rr=[c["keys"]for c in cases if c["keys"]["drug"]==drug and c["keys"]["dose_uM"]==dose]
        ee=[c for c in excluded if c["drug"]==drug and c["dose_uM"]==dose]
        eligibility.append({"condition_id":f"drug={drug}|dose_uM={dose:g}","drug":drug,"dose_uM":dose,
                            "n_annotated_valid_wells":len(rr)+len(ee),"n_ready_wells":len(rr),
                            "n_excluded_wells":len(ee),"n_zero_qc_cells_wells":sum(c["n_treated"]==0 for c in ee),
                            "n_qc_surviving_treated_cells":sum(c["n_treated"]for c in rr+ee),
                            "n_ready_treated_cells":sum(c["n_treated"]for c in rr),
                            "condition_seen_in_training":(drug,dose)in trained_conditions,
                            "status":"ready"if rr else"no_ready_wells"})
    pd.DataFrame(eligibility).to_csv(output/"condition_eligibility.csv",index=False)
    conditions=[]
    for cid in sorted({c["keys"]["condition_id"]for c in cases}):
        well=[c for c in cases if c["keys"]["condition_id"]==cid]
        k=well[0]["keys"]
        counts=Counter(c["keys"]["split"]for c in well)
        keys={"condition_id":cid,"drug":k["drug"],"dose_uM":k["dose_uM"],
              "split":"+".join(sorted(counts)),"seen_in_training":any(c["keys"]["seen_in_training"]for c in well),
              "n_wells":len(well),"n_control":sum(c["keys"]["n_control"]for c in well),
              "n_treated":sum(c["keys"]["n_treated"]for c in well),
              "training_wells":counts["train"],"validation_wells":counts["val"],"test_wells":counts["test"]}
        conditions.append({"keys":keys,**{name:np.mean([c[name]for c in well],axis=0)for name in
                          ["pcontrol","ptreated","axis_delta_genes","observed_delta","decoded_delta",
                           "treated_reconstruction_mse","control_reconstruction_mse"]}})
        item=conditions[-1]
        diffs=np.stack([c["ptreated"]-c["pcontrol"]for c in well])
        item["delta_fraction_well_sd"]=diffs.std(axis=0,ddof=1) if len(well)>1 else np.full_like(diffs[0],np.nan)
        divergences=[divergence(c["pcontrol"],c["ptreated"])for c in well]
        item["mean_well_total_variation"]=np.mean([d[0]for d in divergences],axis=0)
        item["mean_well_js_divergence_base2"]=np.mean([d[1]for d in divergences],axis=0)
        item["mean_well_axis_effect_l2"]=np.mean([np.linalg.norm(c["axis_delta_genes"],axis=1)for c in well],axis=0)
        well_metrics=[metrics(c["observed_delta"],c["decoded_delta"])for c in well]
        item["mean_well_metrics"]={}
        for name in well_metrics[0]:
            values=[m[name]for m in well_metrics if m[name]is not None]
            item["mean_well_metrics"][name]=float(np.mean(values))if values else None
    export_tables(output,"case",cases,labels,axes,states)
    export_tables(output,"condition",conditions,labels,axes,states)
    np.savez_compressed(output/"case_arrays.npz",case_ids=np.asarray([c["keys"]["case_id"]for c in cases]),
                        **{name:np.stack([c[name]for c in cases])for name in
                           ["pcontrol","ptreated","axis_delta_genes","observed_delta","decoded_delta"]})
    np.savez_compressed(output/"condition_gene_deltas.npz",
                        condition_ids=np.asarray([c["keys"]["condition_id"]for c in conditions]),
                        drug=np.asarray([c["keys"]["drug"]for c in conditions]),
                        dose=np.asarray([c["keys"]["dose_uM"]for c in conditions]), genes=genes,symbols=symbols,
                        observed_delta=np.stack([c["observed_delta"]for c in conditions]),
                        decoded_delta=np.stack([c["decoded_delta"]for c in conditions]),
                        axis_gene_delta=np.stack([c["axis_delta_genes"]for c in conditions]))
    old_path=work/"experiment_v2_feedback"/"representation_diagnostic.json"
    consistency={"status":"not_available"}
    if old_path.exists():
        old=json.loads(old_path.read_text()); lookup={c["keys"]["case_id"]:c for c in cases}
        errors={}
        for row in old["cases"]:
            case=lookup[row["case_id"]]
            fresh=delta_metrics(case["observed_delta"],case["decoded_delta"])
            for name,value in fresh.items():
                if value is not None and row.get(name) is not None:
                    errors[name]=max(errors.get(name,0.),abs(float(value)-float(row[name])))
            if row["n_treated"] != case["keys"]["n_treated"] or row["n_control"] != case["keys"]["n_control"]:
                raise ValueError("Prior diagnostic case counts differ")
        if max(errors.values(),default=0)>1e-5:
            raise ValueError(f"Prior diagnostic values differ: {errors}")
        consistency={"status":"passed","n_old_cases":len(old["cases"]),"max_abs_error_by_metric":errors,
                     "reference_path":str(old_path.resolve()),"reference_sha256":fit_lexa.file_sha256(old_path)}
    inputs=[checkpoint_path,prepared/"prepared.npz",prepared/"evaluation.npz",prepared/"metadata.csv",
            fitted/"exact_code_effects.npz",fitted/"labels.json",Path(__file__)]
    report={"status":"complete","created_utc":datetime.now(timezone.utc).isoformat(),
            "posthoc_observed_response_diagnostic":True,"not_prediction":True,
            "contains_training_cells":True,"inputs_sha256":{str(p.resolve()):fit_lexa.file_sha256(p)for p in inputs},
            "samples":{"n_unique_cells":len(cell_ids),"n_genes":n_genes,"axes":axes,"states":states,
                       "archive_input_counts":input_counts,"identical_duplicates_removed":duplicate_count,
                       "cell_splits":cell_meta.split.value_counts().to_dict(),"n_ready_cases":len(cases),
                       "n_low_sample_cases":len(excluded),"n_conditions":len(conditions),
                       "n_valid_annotated_treatment_wells":len(all_treatment_groups),
                       "n_zero_qc_treatment_wells":sum(c["n_treated"]==0 for c in excluded),
                       "n_annotated_conditions":len(eligibility),
                       "invalid_annotated_treatment_wells":sorted(cell_meta_all.loc[cell_meta_all.dose_uM.gt(0)&cell_meta_all.parse_status.ne("ok"),"top_oligo"].dropna().unique().tolist()),
                       "ready_treated_cells":sum(c["keys"]["n_treated"]for c in cases),
                       "ready_unique_control_cells":len(set(cell_meta.loc[np.concatenate([grouped[(c['keys']['drug'],0.,c['keys']['plate'],c['keys']['replicate'])]for c in cases]),'cell_id']))},
            "invariants":{"max_abs_cell_additive_reconstruction_error":cell_error,
                          "max_abs_occupancy_decomposition_error":max(invariant_errors),
                          "identical_duplicate_expression":True,"normalized_occupancy":True,
                          "old_5uM_diagnostic_consistency":consistency},
            "definitions":{"match":"same drug + plate + author replicate zero-dose control",
                           "min_group_cells":args.min_cells,"occupancy":"cell fraction in each categorical axis/state",
                           "total_variation":"0.5 * sum_s abs(p_treated-p_control)",
                           "js_divergence_base2":"0.5*KL_base2(p_control||m)+0.5*KL_base2(p_treated||m), m=(p_control+p_treated)/2",
                           "axis_gene_delta":"sum_s (p_treated-p_control)*exact_axis_state_gene_contribution",
                           "condition_aggregation":"eligible culture wells equally weighted, then metrics computed on averaged vectors; reconstruction MSE averaged equally over wells",
                           "condition_metric_columns":"unprefixed metrics and mean_vector_* describe equally averaged delta vectors; mean_well_* average per-well metrics; total_variation uses mean occupancy whereas mean_well_total_variation averages per-well TV",
                           "delta_fraction_well_sd":"Sample SD (ddof=1) across included well-specific treated-minus-control fractions; NaN for one well",
                           "seen_in_training":"case: treatment cells are in train split; condition: at least one included treatment well is train",
                           "condition_seen_in_training":"drug-dose has any train treatment cells, including wells not passing analysis sample threshold",
                           "n_control_condition":"sum of control group sizes over included wells; same controls can recur across doses",
                           "top100":"100 largest absolute observed delta genes, not statistically significant DEGs",
                           "gene_units":"log1p(CP10K), whole-gene normalization then train-only selection of 2000 features"},
            "limitations":["Observed group comparison only; no same-cell before/after pairing or inferred transition matrix.",
                           "Code IDs are categorical, not numerical activity or dose scales.",
                           "All available doses include train/val; these curves are descriptive, not generalization estimates.",
                           "Author replicate labels are culture-well labels, not established independent biological replicates.",
                           "Cell-level significance and causal mechanisms are not established.",
                           "Condition occupancy/means do not determine full joint single-cell distributions.",
                           "Low-sample groups are excluded and selection can depend on drug response/viability."]}
    report["outputs_sha256"]={p.name:fit_lexa.file_sha256(p)for p in sorted(output.iterdir())if p.is_file()}
    fit_lexa.atomic_json(output/"manifest.json",report)
    print(json.dumps({"output":str(output.resolve()),"samples":report["samples"],
                      "invariants":report["invariants"]},ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work",type=Path,default=Path("runs/pilot_seen_dose"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--min-cells",type=int,default=20)
    args=parser.parse_args()
    if min(args.batch_size,args.threads,args.min_cells)<1:
        raise ValueError("batch-size, threads, min-cells must be positive")
    run(args)


if __name__=="__main__":
    main()
