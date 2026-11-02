"""Numerical tools for a text-only, frozen-LEXA editing agent.

The LLM sees JSON/text; exact signed expression arithmetic stays local.
Observed treated expression is deliberately not part of this interface.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


class Workspace:
    def __init__(self, effects_path, baseline_path, request_path, distribution,
                 output_dir, synthetic=False):
        if distribution not in {"mse_log1p", "nb"}:
            raise ValueError("distribution must be explicit: mse_log1p or nb")
        self.distribution = distribution
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.request = json.loads(Path(request_path).read_text())
        if self.request.get("mode") not in {"functional_edit", "perturbation_prediction"}:
            raise ValueError("request.mode must identify functional_edit or perturbation_prediction")
        self.synthetic = bool(synthetic)
        self.labels = {}
        if self.request.get("state_labels_file"):
            labels_path = Path(self.request["state_labels_file"])
            if not labels_path.is_absolute():
                labels_path = Path(request_path).parent / labels_path
            labels = json.loads(labels_path.read_text())
            if labels.get("effects_sha256") != sha256(effects_path):
                raise ValueError("State labels belong to another frozen codebook")
            self.labels = labels.get("labels", {})
        with np.load(effects_path, allow_pickle=False) as payload:
            self.effects = payload["contributions"].astype(np.float32)
            self.genes = payload["genes"].astype(str)
            self.symbols = payload["gene_symbols"].astype(str) if "gene_symbols" in payload else self.genes.copy()
            self.std = payload["train_gene_std"].astype(np.float32)
            self.occupancy = payload["train_occupancy"].astype(float)
            self.train_codes = payload["train_codes"].copy() if "train_codes" in payload else None
            if "synthetic" in payload and bool(payload["synthetic"].item()) and not self.synthetic:
                raise ValueError("Synthetic effects require --synthetic")
            if "smoke_only" in payload and bool(payload["smoke_only"].item()) and not self.synthetic:
                raise ValueError("Smoke-only effects are not a fitted model; use explicit synthetic/dry-run mode")
        with np.load(baseline_path, allow_pickle=False) as payload:
            self.codes = payload["codes"].copy()
            if "genes" in payload and not np.array_equal(payload["genes"].astype(str), self.genes):
                raise ValueError("Baseline and effects gene orders differ")
            if "synthetic" in payload and bool(payload["synthetic"].item()) and not self.synthetic:
                raise ValueError("Synthetic baseline requires --synthetic")
        if self.effects.ndim != 3:
            raise ValueError("Expected contributions[axis,state,gene]")
        self.axes, self.states, self.gene_count = self.effects.shape
        if self.codes.ndim != 2 or self.codes.shape[1] != self.axes or not len(self.codes):
            raise ValueError("Expected nonempty baseline codes[cell,axis]")
        if not np.issubdtype(self.codes.dtype, np.integer):
            raise ValueError("Baseline codes must be integer")
        if self.codes.min() < 0 or self.codes.max() >= self.states:
            raise ValueError("Baseline code outside vocabulary")
        if self.genes.shape != (self.gene_count,) or len(set(self.genes)) != len(self.genes):
            raise ValueError("Invalid or duplicate gene vocabulary")
        if self.symbols.shape != self.genes.shape:
            raise ValueError("Invalid gene-symbol mapping")
        if self.std.shape != (self.gene_count,) or np.any(self.std < 0):
            raise ValueError("Invalid training gene standard deviations")
        if self.occupancy.shape != (self.axes, self.states) or np.any(self.occupancy < 0):
            raise ValueError("Invalid training occupancy")
        if not all(np.isfinite(v).all() for v in [self.effects, self.std, self.occupancy]):
            raise ValueError("Nonfinite effects/statistics")
        self.scale = np.maximum(self.std, 0.05)
        self.centered = self.effects - self.effects.mean(axis=1, keepdims=True)
        train_codes_file = Path(effects_path).parent / "train_codes.npz"
        if self.train_codes is None and train_codes_file.exists():
            with np.load(train_codes_file, allow_pickle=False) as payload:
                if "genes" in payload and not np.array_equal(payload["genes"].astype(str), self.genes):
                    raise ValueError("Training code vocabulary differs from effects")
                self.train_codes = payload["codes"][:10000].copy()
        if self.train_codes is not None and (self.train_codes.ndim != 2 or self.train_codes.shape[1] != self.axes):
            raise ValueError("Training code axes differ from effects")
        self.baseline_expression = self.decode(self.codes)
        self.calls = 0
        self.provenance = {"effects_sha256": sha256(effects_path),
                           "baseline_sha256": sha256(baseline_path),
                           "request_sha256": sha256(request_path),
                           "distribution": distribution, "synthetic": self.synthetic}

    def decode(self, codes):
        # Accumulate axis by axis to avoid allocating [cells,axes,genes].
        result = np.zeros((len(codes), self.gene_count), dtype=np.float32)
        for axis in range(self.axes):
            result += self.effects[axis, codes[:, axis]]
        if self.distribution == "nb":
            result -= result.max(axis=1, keepdims=True)
            fractions = np.exp(result)
            result = np.log1p(fractions / fractions.sum(axis=1, keepdims=True) * 10000.0)
        return result

    def top_genes(self, vector, n=6):
        values = vector / self.scale
        up = [i for i in np.argsort(-values)[:n] if values[i] > 0]
        down = [i for i in np.argsort(values)[:n] if values[i] < 0]
        def rows(indices):
            return [{"gene": str(self.genes[i]), "symbol": str(self.symbols[i]), "delta_log_expression": float(vector[i]),
                     "delta_over_train_sd": float(values[i])} for i in indices]
        return {"up": rows(up), "down": rows(down)}

    def get_context(self):
        axes = []
        for axis in range(self.axes):
            variation = self.centered[axis].std(axis=0) / self.scale
            genes = [{"gene": str(self.genes[i]), "symbol": str(self.symbols[i])} for i in np.argsort(-variation)[:8]]
            counts = np.bincount(self.codes[:, axis], minlength=self.states)
            axes.append({"axis_id": axis, "varying_genes": genes,
                         "baseline_state_counts": counts.tolist()})
        return {"request": self.request, "axes": axes, "num_states": self.states,
                "baseline_cells": len(self.codes), "provenance": self.provenance,
                "expression_space": "log1p(counts normalized to 10000)",
                "constraints": {"max_edited_axes": 3, "ids_are_zero_based": True,
                                "no_op_allowed": True, "edits_always_start_from_original_baseline": True},
                "support_note": "Training occupancy is marginal; joint support is unknown unless explicitly measured.",
                "scientific_status": "Frozen model edits are hypotheses, not measured treatment effects."}

    def inspect_states(self, axis_id):
        axis = integer(axis_id, "axis_id")
        if not 0 <= axis < self.axes:
            raise ValueError("axis_id outside vocabulary")
        current_mean = self.effects[axis, self.codes[:, axis]].mean(axis=0)
        return {"axis_id": axis, "cards": [
            {"state_id": state,
             "token": f"A{axis}_C{state}",
             "reference": "unweighted within-axis state mean",
             "signed_state_evidence": self.top_genes(self.centered[axis, state]),
             "transition_from_baseline_population": self.top_genes(self.effects[axis, state] - current_mean),
             "training_occupancy_fraction": float(self.occupancy[axis, state]),
             "biological_label": None,
             "language_annotation": self.labels.get(f"A{axis}_C{state}"),
             "evidence_limit": "Decoder-derived expression contribution, not causal regulation; cell-type-specific support unknown."}
            for state in range(self.states)],
            "nb_note": "For NB, these are pre-softmax contributions. evaluate_edits computes actual expression deltas." if self.distribution == "nb" else None}

    def apply_edits(self, edits):
        if not isinstance(edits, list) or len(edits) > 3:
            raise ValueError("edits must be a list with at most three entries")
        result = self.codes.copy()
        seen, descriptions = set(), []
        for item in edits:
            if not isinstance(item, dict) or set(item) - {"axis_id", "to_code", "from_code", "fraction"}:
                raise ValueError("Unexpected edit fields")
            axis = integer(item.get("axis_id"), "axis_id")
            target = integer(item.get("to_code"), "to_code")
            if axis in seen or not 0 <= axis < self.axes or not 0 <= target < self.states:
                raise ValueError("Duplicate axis or out-of-vocabulary edit")
            seen.add(axis)
            source = item.get("from_code")
            if source is not None:
                source = integer(source, "from_code")
                if not 0 <= source < self.states:
                    raise ValueError("from_code outside vocabulary")
            fraction = item.get("fraction", 1.0)
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("fraction must be a finite number in [0,1]")
            eligible = np.flatnonzero(self.codes[:, axis] == source) if source is not None else np.arange(len(result))
            # Same seeded order for every revision/target; fraction is not a causal pairing.
            order = np.random.default_rng(20261026 + axis).permutation(eligible)
            selected = order[:int(np.floor(len(order) * fraction + 0.5))]
            changed = int(np.sum(result[selected, axis] != target))
            result[selected, axis] = target
            descriptions.append({**item, "eligible_cells": len(eligible),
                                 "selected_cells": len(selected), "changed_cells": changed,
                                 "target_training_occupancy": float(self.occupancy[axis, target])})
        return result, descriptions

    def evaluate_edits(self, edits):
        edited, descriptions = self.apply_edits(edits)
        expression = self.decode(edited)
        delta = expression - self.baseline_expression
        if not np.isfinite(delta).all():
            raise ValueError("Nonfinite decoded values")
        self.calls += 1
        artifact = self.output / f"evaluation_{self.calls:03d}.npz"
        if artifact.exists():
            raise FileExistsError("Evaluation output exists; choose a new run directory")
        np.savez_compressed(artifact, genes=self.genes, codes=edited,
                            baseline_mean=self.baseline_expression.mean(axis=0),
                            predicted_mean=expression.mean(axis=0), delta_mean=delta.mean(axis=0),
                            predicted_x=expression,
                            synthetic=np.asarray(self.synthetic))
        panel_scores = []
        gene_index = {gene: i for i, gene in enumerate(self.genes)}
        for panel in self.request.get("panels", []):
            indices = [gene_index[g] for g in panel["genes"] if g in gene_index]
            panel_scores.append({"name": panel["name"], "role": panel.get("role", "unspecified"),
                                 "available_genes": len(indices), "requested_genes": len(panel["genes"]),
                                 "mean_standardized_delta": float((delta.mean(axis=0) / self.scale)[indices].mean()) if indices else None})
        support = {"joint_support": "unknown", "biological_validity": "not_established"}
        if self.train_codes is not None and len(self.train_codes):
            # Bounded diagnostic over at most 64 edited cells and 10000 train cells.
            sample = edited[:64]
            nearest = []
            for row in sample:
                nearest.append(float(np.mean(self.train_codes[:10000] != row, axis=1).min()))
            support["nearest_train_hamming_fraction_mean"] = float(np.mean(nearest))
            support["diagnostic_only"] = True
        result = {"edits": descriptions, "decoded_cells": len(edited),
                  "changed_cells": int(np.any(edited != self.codes, axis=1).sum()),
                  "mean_delta_genes": self.top_genes(delta.mean(axis=0)),
                  "standardized_delta_rms": float(np.sqrt(np.mean(np.square(delta / self.scale)))),
                  "negative_decoded_values_fraction": float(np.mean(expression < 0)),
                  "panel_scores": panel_scores, "support": support,
                  "artifact": str(artifact.resolve()), "provenance": self.provenance,
                  "evaluation_scope": "Code legality and frozen-model expression change only; no observed treatment data accessed."}
        (self.output / f"evaluation_{self.calls:03d}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result

    def export_cards(self, path):
        with Path(path).open("w") as handle:
            for axis in range(self.axes):
                for card in self.inspect_states(axis)["cards"]:
                    handle.write(json.dumps({"axis_id": axis, **card}, ensure_ascii=False) + "\n")
