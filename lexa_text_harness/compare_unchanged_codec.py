#!/usr/bin/env python3
"""Compare unchanged control decoding with observed treated population means."""
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
    source=folder/'cell_codes.npz';manifest=json.loads((folder/'manifest.json').read_text())
    assert sha(source)==manifest['outputs_sha256']['cell_codes.npz']
    cases=[c for c in json.loads((work/'prepared/cases.json').read_text())['cases'] if c['status']=='ready']
    with np.load(source,allow_pickle=False) as z:
        ids=z['cell_ids'].astype(str);x=z['observed_x'];xr=z['reconstructed_x'];genes=z['genes'].astype(str)
    lookup={v:i for i,v in enumerate(ids)}
    assert len(lookup)==len(ids)
    run=work/'experiment_v3_hierarchy_fixed'
    seal=json.loads((run/'prediction_seal.json').read_text())
    assert seal['status']=='all_predictions_complete'
    methods={
        'observed_control_unchanged':'mean(control)',
        'decoded_control_unchanged':'mean(D(E(control)))',
        'decoded_true_treated':'mean(D(E(treated)))',
        'agent_direct_decoded':'mean(D(E(control))) + predicted_decoder_delta = mean(D(predicted_codes))',
        'agent_observed_control_anchored':'mean(control) + predicted_decoder_delta; previous prediction-table convention',
        'true_treated_observed_control_anchored':'mean(control) + mean(D(E(treated))) - mean(D(E(control))); previous codec delta convention'}
    rows=[]
    for c in cases:
        ti=[lookup[v] for v in c['treated_cell_ids']];ci=[lookup[v] for v in c['control_cell_ids']]
        ym=x[ti].mean(0,dtype=float);yrm=xr[ti].mean(0,dtype=float)
        bm=x[ci].mean(0,dtype=float);brm=xr[ci].mean(0,dtype=float)
        slug=hashlib.sha256(c['case_id'].encode()).hexdigest()[:12]
        artifact=run/'predictions'/slug/'joint.npz'
        assert sha(artifact)==seal['files'][str(artifact.relative_to(run))]
        with np.load(artifact,allow_pickle=False) as z:
            assert np.array_equal(z['genes'].astype(str),genes)
            delta=z['delta_mean'].copy()
        predictions={'observed_control_unchanged':bm,'decoded_control_unchanged':brm,
            'decoded_true_treated':yrm,'agent_direct_decoded':brm+delta,
            'agent_observed_control_anchored':bm+delta,
            'true_treated_observed_control_anchored':bm+yrm-brm}
        for method,prediction in predictions.items():
            rows.append({'case_id':c['case_id'],'drug':c['drug'],'method':method,
                'n_treated':len(ti),'n_control':len(ci),
                'rmse_to_observed_treated_mean':float(np.sqrt(np.mean((prediction-ym)**2)))})
    table=pd.DataFrame(rows)
    by_drug=table.groupby(['method','drug'],as_index=False)['rmse_to_observed_treated_mean'].mean()
    summary=by_drug.groupby('method')['rmse_to_observed_treated_mean'].mean().to_dict()
    old=json.loads((folder/'direct_codec_error.json').read_text())['equal_drug_mean_of_well_metrics']
    assert abs(summary['decoded_true_treated']-old['treated_population_mean_rmse'])<1e-12
    assert abs(summary['true_treated_observed_control_anchored']-old['response_delta_rmse'])<1e-12
    previous=pd.read_csv(run/'scores_by_well.csv').query('method == "hierarchical_joint"').set_index('case_id')
    current=table.query('method == "agent_observed_control_anchored"').set_index('case_id')
    discrepancy=float(np.max(np.abs(current.rmse_to_observed_treated_mean-previous.delta_rmse)))
    assert discrepancy<2e-6 and len(cases)==16
    report={'status':'complete','n_wells':16,'n_genes':len(genes),'n_treated':sum(c['n_treated'] for c in cases),
        'units':'log1p(CP10K)','target_for_every_row':'Observed treated population mean for the same matched well',
        'aggregation':'Per-well RMSE across2000genes, equal wells per drug, then equal four drugs',
        'methods':methods,'overall_rmse':summary,'by_drug':by_drug.to_dict(orient='records'),
        'direct_agent_rmse_reduction_vs_decoded_unchanged_pct':100*(1-summary['agent_direct_decoded']/summary['decoded_control_unchanged']),
        'verification_max_agent_anchored_difference_from_prior_float32_score':discrepancy,
        'source_sha256':{'cell_codes.npz':sha(source),'cases.json':sha(work/'prepared/cases.json'),
                         'prediction_seal.json':sha(run/'prediction_seal.json')},
        'limitations':['Control and treated cells are unpaired; comparison uses group means, not paired single cells.',
            'Direct decoder outputs and observed-control-anchored outputs have different reconstruction bias.',
            'Encoding true treated cells is a diagnostic using the answer, not a perturbation predictor or strict error bound.']}
    table.to_csv(folder/'unchanged_control_comparison_by_well.csv',index=False)
    by_drug.to_csv(folder/'unchanged_control_comparison_by_drug.csv',index=False)
    (folder/'unchanged_control_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'overall':summary,'relative_gain_pct':report['direct_agent_rmse_reduction_vs_decoded_unchanged_pct'],
                      'per_drug':by_drug.to_dict(orient='records')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
