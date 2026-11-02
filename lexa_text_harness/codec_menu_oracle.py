#!/usr/bin/env python3
"""Answer-using two-start coordinate search of the existing seven-axis-candidate menu.
This is a feasible local search diagnostic, not a globally certified optimum or prediction.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import hashlib
import numpy as np
import pandas as pd


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parent/'runs/pilot_seen_dose')
    args=parser.parse_args();work=args.work.resolve();out=work/'codec_capacity_oracle'
    out.mkdir(exist_ok=True)
    if (out/'menu_results.json').exists():raise ValueError('Already completed')
    run=work/'experiment_v3_hierarchy_fixed'
    m=json.loads((run/'inputs/manifest.json').read_text())
    with np.load(work/'codec_changes/case_arrays.npz') as z:
        truths={str(c):z['observed_delta'][i].copy() for i,c in enumerate(z['case_ids'])}
    rows=[];choices=[];deltas=[]
    for case in m['cases']:
        assert sha(case['npz'])==m['output_sha256'][case['npz']]
        with np.load(case['npz']) as z:
            effect=z['candidate_delta_genes'].copy();names=z['candidate_names'].astype(str).tolist()
        target=truths[case['case_id']];best=None;starts=[]
        for name in ['nearest','interpolate']:
            choice=np.full(32,names.index(name),dtype=int)
            prediction=effect[np.arange(32),choice].sum(0)
            initial=float(np.mean((prediction-target)**2))
            moves=0
            for sweep in range(20):
                changed=0
                for axis in range(32):
                    others=prediction-effect[axis,choice[axis]]
                    residual=others[None,:]+effect[axis]-target[None,:]
                    loss=np.mean(residual**2,axis=1)
                    k=int(loss.argmin())
                    if loss[k]<loss[choice[axis]]-1e-14:
                        prediction=others+effect[axis,k];choice[axis]=k;changed+=1;moves+=1
                if changed==0:break
            mse=float(np.mean((prediction-target)**2))
            starts.append({'start':name,'initial_rmse':float(np.sqrt(initial)),'final_rmse':float(np.sqrt(mse)),'sweeps':sweep+1,'moves':moves})
            if best is None or mse<best[0]:best=(mse,choice.copy(),prediction.copy(),name)
        mse,choice,prediction,name=best
        assert np.isfinite(mse)
        rows.append({'case_id':case['case_id'],'drug':case['drug'],'menu_feasible_rmse':float(np.sqrt(mse)),
            'best_start':name,'candidate_counts':{n:int((choice==i).sum()) for i,n in enumerate(names)},'start_details':starts})
        choices.append(choice);deltas.append(prediction)
    table=pd.DataFrame([{k:v for k,v in r.items() if k not in ('candidate_counts','start_details')} for r in rows])
    table.to_csv(out/'menu_by_well.csv',index=False)
    bydrug=table.groupby('drug').menu_feasible_rmse.mean()
    result={'status':'complete','uses_target_answers':True,'is_prediction':False,'is_global_bound':False,
        'created_asia_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'method':'Coordinate descent over32axes7candidates, nearest/interpolate starts,20sweeps maximum, strict improving moves, retain lower final objective',
        'mean_equal_well_rmse':float(bydrug.mean()),'by_drug_rmse':bydrug.to_dict(),'cases':rows,
        'limitation':'Truth-selected local optimum of fixed candidate menu; cannot be used as predictive performance or a certified minimum.',
        'source_sha256':{'inputs_manifest':sha(run/'inputs/manifest.json'),'truth_case_arrays':sha(work/'codec_changes/case_arrays.npz'),'script':sha(__file__)}}
    np.savez_compressed(out/'menu_fitted_arrays.npz',case_ids=np.asarray([c['case_id'] for c in m['cases']]),choices=np.stack(choices),fitted_delta=np.stack(deltas),candidate_ids=np.asarray(names))
    (out/'menu_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'mean_rmse':result['mean_equal_well_rmse'],'by_drug':result['by_drug_rmse']},indent=2))


if __name__=='__main__':main()
