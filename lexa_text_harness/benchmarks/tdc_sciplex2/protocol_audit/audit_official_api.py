#!/usr/bin/env python3
"""Audit unmodified published API methods using synthetic metadata/expression only.
No dataset is loaded, no model is trained, and no HTTP request is made.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / 'vendor' / 'pytdc-1.1.15' / 'tdc'

def extract(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[method_name]

split_path = VENDOR / 'multi_pred' / 'perturboutcome.py'
score_path = VENDOR / 'benchmark_group' / 'counterfactual_group.py'
split = extract(split_path, 'PerturbOutcome', 'get_cellline_split', {'train_test_split': train_test_split})
obs = pd.DataFrame({'cell_line': ['A549'] * 500,
                    'perturbation': np.repeat(['control', 'BMS', 'Dex', 'Nutlin', 'SAHA'], 100),
                    'ncounts': np.arange(500)})
fake_loader = SimpleNamespace(get_data=lambda: SimpleNamespace(obs=obs))
seen = split(fake_loader)
results = {'package': 'PyTDC 1.1.15', 'data': 'synthetic only, 100 cells per control/drug',
           'source_sha256': {str(p.relative_to(VENDOR)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [split_path, score_path]},
           'default_seen': {k: len(v) if hasattr(v, '__len__') else v for k, v in seen['A549'].items()},
           'returned_train_columns': list(seen['A549']['train'].columns)}
assert results['default_seen']['test'] == results['default_seen']['dev'] == 0
try:
    split(fake_loader, split_to_unseen=True)
except ValueError as exc:
    results['default_unseen'] = {'error_type': type(exc).__name__, 'error': str(exc)}
else:
    raise AssertionError('Expected four-drug default cold split to fail')

score = extract(score_path, 'CounterfactualGroup', 'evaluate', {'pd': pd, 'np': np})
drug_cols = ['ncounts', 'celltype', 'cell_line', 'cancer', 'disease', 'tissue_type',
             'perturbation', 'perturbation_type', 'ngenes']
truth = pd.DataFrame({'gene1': [1., 1.], 'gene2': [2., 2.],
                      **{c: ['metadata', 'metadata'] for c in drug_cols}})
truth['ncounts'] = 1000.
truth['ngenes'] = 400.
pred = truth.copy()
pred[['gene1', 'gene2']] = 0.
fake_group = SimpleNamespace(get_test=lambda: {'A549': truth}, is_drug=True, _DRUG_COLS=drug_cols)
results['numeric_metadata_drop_probe'] = {
    'official_evaluate': float(score(fake_group, {'A549': pred})),
    'expression_only_r2': float(r2_score(truth[['gene1','gene2']].mean(), pred[['gene1','gene2']].mean())),
    'interpretation': 'Official ignored drop() retains ncounts/ngenes, contaminating score.'}
results['caution'] = 'These are API regression probes, not biological benchmark scores.'
(HERE / 'audit_results.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
