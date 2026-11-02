"""Inspect only structure and cell/gene annotations of verified scPerturb H5AD.

Does not load expression values, compute differential expression, or score a model.
The source file is not assumed identical to the inaccessible TDC Dataverse copy.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
NAME = "SrivatsanTrapnell2020_sciplex2.h5ad"
EXPECTED_SIZE = 145178504
EXPECTED_MD5 = "12a48293a166a859f994551653a2ad8c"


def plain(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def describe(node):
    result = {"kind": "dataset" if isinstance(node, h5py.Dataset) else "group",
              "attrs": {k: plain(v) for k, v in node.attrs.items()}}
    if isinstance(node, h5py.Dataset):
        result.update(shape=list(node.shape), dtype=str(node.dtype),
                      compression=node.compression)
    else:
        result["children"] = {
            k: {"kind": "dataset" if isinstance(v, h5py.Dataset) else "group",
                **({"shape": list(v.shape), "dtype": str(v.dtype)}
                   if isinstance(v, h5py.Dataset) else {})}
            for k, v in node.items()
        }
    return result


def values(node):
    if isinstance(node, h5py.Dataset):
        return np.asarray([plain(v) for v in node[()]], dtype=object)
    if "codes" in node and "categories" in node:
        cats = values(node["categories"])
        return np.asarray([None if c < 0 else cats[c]
                           for c in node["codes"][()]], dtype=object)
    if "values" in node and "mask" in node:
        vals = values(node["values"])
        vals[node["mask"][()].astype(bool)] = None
        return vals
    raise ValueError(f"Unsupported annotation representation: {node.name}")


def inspect():
    path = ROOT / "data" / NAME
    assert path.stat().st_size == EXPECTED_SIZE, "Full source file required"
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    assert md5.hexdigest() == EXPECTED_MD5, "Official source MD5 mismatch"
    report = {"source_file": str(path), "size_bytes": path.stat().st_size,
              "md5": md5.hexdigest(), "sha256": sha256.hexdigest(),
              "source_md5_verified": True,
              "tdc_dataverse_file_id": 9845394,
              "tdc_dataverse_byte_equivalence": "unverified; access returned HTTP403",
              "expression_values_loaded": False,
              "model_scoring_performed": False}
    with h5py.File(path, "r") as file:
        report["root"] = describe(file)
        obs = file["obs"]
        var = file["var"]
        obs_index = plain(obs.attrs.get("_index", "_index"))
        var_index = plain(var.attrs.get("_index", "_index"))
        report["n_obs"] = len(obs[obs_index])
        report["n_vars"] = len(var[var_index])
        report["obs_columns"] = [k for k in obs if k != obs_index]
        report["obs_annotation_summary"] = {}
        for key in report["obs_columns"]:
            vals = values(obs[key])
            counts = Counter(str(v) for v in vals)
            summary = {"representation": describe(obs[key]),
                       "n_unique": len(counts), "n_values": len(vals)}
            if len(counts) <= 100:
                summary["counts"] = dict(sorted(counts.items()))
            report["obs_annotation_summary"][key] = summary
        if "perturbation" in obs and "dose_value" in obs:
            pairs = Counter(zip(map(str, values(obs["perturbation"])),
                                map(str, values(obs["dose_value"]))))
            report["perturbation_by_dose_counts"] = [
                {"perturbation": p, "dose_value": d, "n_obs": n}
                for (p, d), n in sorted(pairs.items())]
            report["control_annotation_caveat"] = (
                "Explicit perturbation='control' and drug-labeled dose_value='0' "
                "are separate observed annotations. No relabeling was performed. "
                "Dose units are not encoded in the listed obs columns.")
        report["var_index_name"] = var_index
        report["var_index_examples"] = [plain(v) for v in var[var_index][:10]]
        report["var_index_n_unique"] = len(set(map(str, values(var[var_index]))))
        report["var_columns"] = [k for k in var if k != var_index]
        report["var_annotation_summary"] = {}
        for key in report["var_columns"]:
            vals = values(var[key])
            report["var_annotation_summary"][key] = {
                "representation": describe(var[key]),
                "examples": [plain(v) for v in vals[:10]],
                "n_unique": len(set(str(v) for v in vals))}
        report["X"] = describe(file["X"])
        report["layers"] = {k: describe(v) for k, v in file.get("layers", {}).items()}
        report["raw"] = describe(file["raw"]) if "raw" in file else None
        report["uns_keys"] = list(file.get("uns", {}).keys())
    dest = ROOT / "data" / "source_metadata_report.json"
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"report": str(dest), "n_obs": report["n_obs"],
                      "n_vars": report["n_vars"], "obs_columns": report["obs_columns"],
                      "md5_verified": report["source_md5_verified"]}, indent=2))
    return report


if __name__ == "__main__":
    inspect()
