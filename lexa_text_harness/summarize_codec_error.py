#!/usr/bin/env python3
"""Summarize direct frozen encode/decode error on the same 16 ready 5uM wells."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parent/'runs/pilot_seen_dose')
    args=parser.parse_args();work=args.work;folder=work/'codec_changes'
    manifest=json.loads((folder/'manifest.json').read_text())
    path=folder/'cell_codes.npz'
    if sha(path)!=manifest['outputs_sha256']['cell_codes.npz']:
        raise ValueError('Previously verified codec reconstruction arrays changed')
    cases=[c for c in json.loads((work/'prepared/cases.json').read_text())['cases'] if c['status']=='ready']
    with np.load(path,allow_pickle=False) as z:
        ids=z['cell_ids'].astype(str);x=z['observed_x'];xr=z['reconstructed_x'];genes=z['genes'].astype(str)
    lookup={v:i for i,v in enumerate(ids)}
    if len(lookup)!=len(ids):raise ValueError('Duplicate cells')
    rows=[];treated_unique=set();control_unique=set()
    for c in cases:
        ti=[lookup[v] for v in c['treated_cell_ids']];ci=[lookup[v] for v in c['control_cell_ids']]
        treated_unique.update(ti);control_unique.update(ci)
        y=x[ti].astype(float);yr=xr[ti].astype(float)
        b=x[ci].astype(float);br=xr[ci].astype(float)
        ey=yr.mean(0)-y.mean(0);eb=br.mean(0)-b.mean(0)
        delta=y.mean(0)-b.mean(0);delta_hat=yr.mean(0)-br.mean(0)
        rmse=lambda a:float(np.sqrt(np.mean(a*a)))
        rows.append({'case_id':c['case_id'],'drug':c['drug'],'n_treated':len(ti),'n_control':len(ci),
            'response_delta_rmse':rmse(delta_hat-delta),
            'treated_population_mean_rmse':rmse(ey),'control_population_mean_rmse':rmse(eb),
            'treated_cell_matrix_rmse':rmse(yr-y),'control_cell_matrix_rmse':rmse(br-b),
            'response_delta_pearson':float(np.corrcoef(delta,delta_hat)[0,1]),
            'response_norm_retained':float(np.linalg.norm(delta_hat)/np.linalg.norm(delta))})
    table=pd.DataFrame(rows);metric_cols=[c for c in table if c.endswith(('rmse','pearson','retained'))]
    by_drug=table.groupby('drug')[metric_cols].mean()
    summary=by_drug.mean().to_dict()
    pooled={}
    for name,indices in [('treated',sorted(treated_unique)),('control',sorted(control_unique))]:
        err=xr[indices].astype(float)-x[indices].astype(float)
        pooled[name]={'n_unique_cells':len(indices),'cell_gene_pooled_rmse':float(np.sqrt(np.mean(err**2)))}
    original=json.loads((work/'experiment_v2_feedback/representation_diagnostic.json').read_text())
    reference={c['case_id']:c['delta_rmse'] for c in original['cases']}
    max_error=max(abs(r['response_delta_rmse']-reference[r['case_id']]) for r in rows)
    assert max_error<1e-12 and len(rows)==16 and len(treated_unique)==786
    report={'status':'complete','uses_true_treatment':'direct frozen reconstruction; no LLM or drug prediction',
        'units':'log1p(CP10K), 2000 training-selected genes',
        'n_wells':len(rows),'n_drugs':len(by_drug),'n_genes':len(genes),'n_unique_cells':pooled,
        'aggregation':'RMSE within each well; equal wells per drug then equal drugs. Pooled cell-gene RMSE reported separately.',
        'equal_drug_mean_of_well_metrics':summary,'per_drug_mean_of_well_metrics':by_drug.to_dict(orient='index'),
        'definitions':{'response_delta_rmse':'RMSE((mean D(E(treated))-mean D(E(control)))-(mean treated-mean control))',
            'treated_population_mean_rmse':'RMSE(mean D(E(treated))-mean treated)',
            'treated_cell_matrix_rmse':'RMSE over cells and genes within one treated well, then well/drug averaged',
            'cell_gene_pooled_rmse':'sqrt(mean squared reconstruction error over all unique selected cells and genes))'},
        'max_difference_from_previous_verified_delta_rmse':max_error,
        'source_sha256':{'cell_codes.npz':sha(path),'cases.json':sha(work/'prepared/cases.json')},
        'limitations':['True-answer encoding is a representation diagnostic, not a predictor or strict attainable-error bound.',
            'RMSEs are different quantities and cannot be subtracted to isolate language-model error.',
            'Discrete quantization, encoder and decoder losses are combined; their separate contributions are not identified.']}
    table.to_csv(folder/'direct_codec_error_by_well.csv',index=False)
    (folder/'direct_codec_error.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summary':summary,'pooled':pooled,'per_drug':by_drug.to_dict(orient='index')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
