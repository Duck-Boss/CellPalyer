#!/usr/bin/env python3
"""Answer-using capacity diagnostic for continuous group code occupancies.

Not a perturbation predictor. The observed target response is explicitly fitted.
Only the fixed additive decoder and product-of-simplex marginals constrain the
fit; joint biological cell support is NOT imposed.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(values):
    shape=values.shape
    flat=values.reshape(-1,shape[-1])
    s=np.sort(flat,axis=1)[:,::-1]
    c=np.cumsum(s,axis=1)-1
    valid=s-c/np.arange(1,shape[-1]+1)>0
    k=valid.sum(axis=1)-1
    threshold=c[np.arange(len(flat)),k]/(k+1)
    return np.maximum(flat-threshold[:,None],0).reshape(shape)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parent/'runs/pilot_seen_dose')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-iterations',type=int,default=10000)
    parser.add_argument('--gap-tolerance',type=float,default=1e-8)
    parser.add_argument('--max-seconds',type=float,default=150)
    args=parser.parse_args();work=args.work.resolve();output=args.output or work/'codec_capacity_oracle'
    if (output/'results.json').exists():raise ValueError('Output complete; choose new directory')
    output.mkdir(parents=True,exist_ok=True)
    arraypath=work/'codec_changes/case_arrays.npz';effectpath=work/'lexa/exact_code_effects.npz'
    manifest=json.loads((work/'codec_changes/manifest.json').read_text())
    assert sha(arraypath)==manifest['outputs_sha256']['case_arrays.npz']
    assert sha(effectpath)==manifest['inputs_sha256'][str(effectpath)]
    cases=sorted([c for c in json.loads((work/'prepared/cases.json').read_text())['cases'] if c['status']=='ready'],key=lambda c:c['case_id'])
    with np.load(arraypath,allow_pickle=False) as z:
        lookup={str(x):i for i,x in enumerate(z['case_ids'])};ix=[lookup[c['case_id']] for c in cases]
        p0=z['pcontrol'][ix].astype(float);qt=z['ptreated'][ix].astype(float)
        target_delta=z['observed_delta'][ix].astype(float);old_delta=z['decoded_delta'][ix].astype(float)
    with np.load(effectpath,allow_pickle=False) as z:
        effect=z['contributions'].astype(float);genes=z['genes'].astype(str)
    n,a,s=p0.shape;g=len(genes)
    assert (n,a,s,g)==(16,32,16,2000)
    assert np.allclose(qt.sum(2),1) and np.allclose(p0.sum(2),1)
    centered=effect-effect.mean(axis=1,keepdims=True)
    U=centered.reshape(a*s,g)
    target=target_delta+p0.reshape(n,-1)@U
    H=U@U.T/g;c=target@U.T/g
    const=np.sum(target**2,axis=1)/g
    eigen_max=float(np.linalg.eigvalsh(H)[-1]);L=2*eigen_max
    q=qt.reshape(n,-1).copy();y=q.copy();t=1.
    def objective(values):
        return np.maximum(np.sum((values@H)*values,axis=1)-2*np.sum(values*c,axis=1)+const,0)
    best=q.copy();best_f=objective(q);start=time.monotonic();converged=False
    for iteration in range(1,args.max_iterations+1):
        grad=2*(y@H-c)
        qnew=project((y-grad/L).reshape(n,a,s)).reshape(n,-1)
        f=objective(qnew);better=f<best_f
        best[better]=qnew[better];best_f[better]=f[better]
        tnew=(1+np.sqrt(1+4*t*t))/2
        y=qnew+(t-1)/tnew*(qnew-q);q=qnew;t=tnew
        if iteration%50==0 or iteration==args.max_iterations:
            grad=2*(best@H-c)
            gap=np.sum(best*grad,axis=1)-grad.reshape(n,a,s).min(axis=2).sum(axis=1)
            gap=np.maximum(gap,0)
            if iteration%500==0:
                print(json.dumps({'iteration':iteration,'mean_rmse':float(np.sqrt(best_f).mean()),'max_gap':float(gap.max()),'elapsed':time.monotonic()-start}),flush=True)
            if gap.max()<=args.gap_tolerance:
                converged=True;break
            if time.monotonic()-start>=args.max_seconds:break
    qfit=best.reshape(n,a,s)
    predicted=(best-p0.reshape(n,-1))@U
    encoded=(qt.reshape(n,-1)-p0.reshape(n,-1))@U
    assert np.max(np.abs(encoded-old_delta))<1e-5
    residual=predicted-target_delta
    mse=np.mean(residual**2,axis=1)
    gradient=2*(residual@U.T)/g
    gap=np.maximum(np.sum(best*gradient,axis=1)-gradient.reshape(n,a,s).min(2).sum(1),0)
    lower=np.maximum(0,mse-gap)
    assert qfit.min()>=-1e-12 and np.max(np.abs(qfit.sum(2)-1))<1e-10
    initial=np.mean((encoded-target_delta)**2,axis=1)
    assert np.all(mse<=initial+1e-12)
    rows=[]
    for i,case in enumerate(cases):
        rows.append({'case_id':case['case_id'],'drug':case['drug'],'n_treated':case['n_treated'],
            'encoded_true_response_rmse':float(np.sqrt(initial[i])),
            'fitted_feasible_rmse':float(np.sqrt(mse[i])),
            'relaxed_optimum_rmse_lower_bound':float(np.sqrt(lower[i])),
            'mse_frank_wolfe_gap':float(gap[i]),
            'occupancy_tv_vs_true_encoded_mean':float(.5*np.abs(qfit[i]-qt[i]).sum(1).mean()),
            'gap_tolerance_met':bool(gap[i]<=args.gap_tolerance)})
    table=pd.DataFrame(rows);table.to_csv(output/'by_well.csv',index=False)
    metrics=['encoded_true_response_rmse','fitted_feasible_rmse','relaxed_optimum_rmse_lower_bound','mse_frank_wolfe_gap','occupancy_tv_vs_true_encoded_mean']
    condition=table.groupby('drug')[metrics].mean();condition.to_csv(output/'by_drug.csv')
    np.savez_compressed(output/'fitted_arrays.npz',case_ids=np.asarray([c['case_id'] for c in cases]),p0=p0,
        true_encoded_q=qt,fitted_q=qfit,observed_delta=target_delta,fitted_delta=predicted,genes=genes)
    result={'status':'complete','created_asia_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'uses_target_answers':True,'is_prediction':False,'n_cases':n,'n_genes':g,'n_treated':sum(c['n_treated'] for c in cases),
        'method':'FISTA projected onto product of32 probability simplexes; initialized at actual treated encoded occupancies; retain lowest feasible objective per case',
        'objective':'mean_g((sum_as (q_as-p0_as)*u_asg)-observed_delta_g)^2',
        'constraints':'q_as>=0 and sum_s q_as=1, continuous population fractions; no joint-support or finite-cell-count constraint',
        'iterations':iteration,'elapsed_seconds':time.monotonic()-start,'converged_all':bool(np.all(gap<=args.gap_tolerance)),
        'gap_tolerance':args.gap_tolerance,'lipschitz_constant':L,
        'mean_equal_well_metrics':condition.mean().to_dict(),'per_drug':condition.to_dict(orient='index'),
        'certificate':'For convex differentiable f on product simplexes, gap=max_v gradf(q) dot(q-v)=q dot gradf(q)-min_v v dot gradf(q), so f(q)-gap <= f* <= f(q); numerical floating-point gap is a convergence diagnostic, not exact-arithmetic proof.',
        'units':'log1p(CP10K); individual well RMSE then equal wells/drugs',
        'limitations':['This fits known true5uM answers and cannot be reported as perturbation prediction.',
            'Optimum applies only to relaxed population marginals and mean expression, not real biological joint code profiles.',
            'No extrapolation/generalization or causal claim; no drug-response model learned.',
            'True encoded occupancies need not minimize this population response objective.'],
        'source_sha256':{'case_arrays.npz':sha(arraypath),'exact_code_effects.npz':sha(effectpath),'script':sha(__file__)},
        'outputs_sha256':{name:sha(output/name) for name in ['by_well.csv','by_drug.csv','fitted_arrays.npz']}}
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'summary':result['mean_equal_well_metrics'],'iterations':iteration,'elapsed':result['elapsed_seconds'],'converged_all':result['converged_all']},indent=2),flush=True)


if __name__=='__main__':main()
