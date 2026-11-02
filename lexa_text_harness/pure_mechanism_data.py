#!/usr/bin/env python3
"""Build control-only inputs for mechanism-and-code-semantic prediction.

The allowlist of read files deliberately excludes all treatment response
libraries, evaluation arrays, inferred dose curves and answer-fitting outputs.
Drug mechanisms are supplied as separately versioned primary-knowledge cards.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import re

import numpy as np


VERSION = "pure-mechanism-inputs-v1"
PUBLIC_ID_SALT = "pure-mechanism-v4-fixedpublicsalt20261027:"
ANON_SEED = 20261027
DRUG_IDENTITIES = re.compile(r"\b(?:BMS(?:[- ]?345541)?|Dex|Dexamethasone|Nutlin(?:[- ]?3a?)?|SAHA|Vorinostat)\b", re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def public_id(case_id):
    return "P"+hashlib.sha256((PUBLIC_ID_SALT+case_id).encode()).hexdigest()[:12]


def anonymous_genes(n):
    # Random labels, not lexicographic ranks or gene-derived hashes.
    order=np.random.default_rng(ANON_SEED).permutation(n)
    return np.asarray([f"G{i:04d}" for i in order])


def occupancy(codes, n_states):
    codes=np.asarray(codes)
    if codes.ndim!=2 or not len(codes) or not np.issubdtype(codes.dtype,np.integer):
        raise ValueError("Expected a nonempty integer cell-by-axis code matrix")
    if codes.min()<0 or codes.max()>=n_states:
        raise ValueError("Control code outside vocabulary")
    return np.stack([np.bincount(codes[:,a],minlength=n_states)/len(codes) for a in range(codes.shape[1])])


def gene_rows(values, genes, symbols, anonymous_ids, std, sign=1, count=2):
    values=np.asarray(values,dtype=float)
    scaled=values/np.maximum(std,.05)
    indices=np.flatnonzero(sign*scaled>1e-8)
    indices=indices[np.argsort(-sign*scaled[indices],kind="stable")][:count]
    return [{"gene_id":str(genes[g]),"symbol":str(symbols[g]),"anonymous_id":str(anonymous_ids[g]),
             "delta":round(float(values[g]),6),"scaled_delta":round(float(scaled[g]),6)} for g in indices]


def make_axis_cards(effects,mask,genes,symbols,std,train_occupancy,labels,annotations):
    centered=effects-effects.mean(axis=1,keepdims=True)
    importance=np.sqrt(np.mean(centered**2,axis=1))/np.maximum(std,.05)
    anon=anonymous_genes(len(genes));cards=[]
    for axis in range(len(effects)):
        varying=np.argsort(-importance[axis],kind="stable")[:6]
        mask_indices=np.argsort(-mask[axis],kind="stable")[:6]
        states=[]
        for state in range(effects.shape[1]):
            label=labels[f"A{axis}_C{state}"]
            states.append({"state_id":state,"name":label["name"],"summary":label["summary"],
                "evidence_ref":f"A{axis}/C{state}",
                "train_occupancy":round(float(train_occupancy[axis,state]),6),
                "positive_genes":gene_rows(centered[axis,state],genes,symbols,anon,std,1),
                "negative_genes":gene_rows(centered[axis,state],genes,symbols,anon,std,-1)})
        annotation=annotations.get(axis)
        card={"axis_id":axis,"evidence_ref":f"A{axis}",
              "scope":"/".join(symbols[varying[:4]])+" associated expression program",
              "top_varying_genes":[{"gene_id":str(genes[g]),"symbol":str(symbols[g]),"anonymous_id":str(anon[g]),
                                    "rms_over_train_sd":round(float(importance[axis,g]),6)} for g in varying],
              "top_mask_genes":[{"gene_id":str(genes[g]),"symbol":str(symbols[g]),"anonymous_id":str(anon[g]),
                                 "responsibility":round(float(mask[axis,g]),6)} for g in mask_indices],
              "state_reference":"uniform mean contribution over this axis's sixteen states",
              "states":states,"annotation_status":"Provisional gene-grounded language, no formal functional enrichment"}
        if annotation:
            card["language_annotation"]=annotation
        # State/axis cards are drug-agnostic descriptions of frozen effects.
        if DRUG_IDENTITIES.search(json.dumps(card,ensure_ascii=False)):
            raise ValueError("A supposedly drug-agnostic axis/state card contains a drug identity")
        cards.append(card)
    return cards


def _mechanisms(document):
    if isinstance(document,dict):
        entries=document.get("drugs",document.get("mechanisms",document))
    else:
        entries=document
    if isinstance(entries,list):
        result={v["drug"]:v for v in entries}
        if len(result)!=len(entries):raise ValueError("Duplicate mechanism drug")
        return result
    if not isinstance(entries,dict):raise ValueError("Expected mechanism cards keyed by drug")
    return entries


def _control_summary(values, genes, symbols, anon):
    mean=values.mean(axis=0)
    indices=np.argsort(-mean,kind="stable")[:12]
    return {"n_control":len(values),"normalization":"log1p(CP10K)",
        "most_abundant_genes":[{"gene_id":str(genes[g]),"symbol":str(symbols[g]),"anonymous_id":str(anon[g]),
            "mean_expression":round(float(mean[g]),5)} for g in indices],
        "limitation":"Untreated population only; expression does not establish protein activity or genotype"}


def prepare_inputs(work,output,mechanism_path):
    work=Path(work).resolve();output=Path(output).resolve();mechanism_path=Path(mechanism_path).resolve()
    paths={"effects":work/'lexa/exact_code_effects.npz',"baseline":work/'lexa/baseline.npz',
           "state_labels":work/'lexa/labels.json',"cases":work/'prepared/cases.json',
           "axis_annotations":work/'experiment_v3_hierarchy_fixed/axis_annotations.json',
           "mechanisms":mechanism_path}
    # This explicit allowlist is the complete data dependency set.
    provenance={"version":VERSION,"work":str(work),"source_paths":{k:str(v) for k,v in paths.items()},
        "source_sha256":{k:sha(v) for k,v in paths.items()},"implementation_sha256":sha(__file__),
        "anonymous_gene_seed":ANON_SEED,"condition_response_retrieval":False,"heldout_expression_accessed":False}
    manifest_path=output/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['provenance']!=provenance:raise ValueError("Existing pure-mechanism inputs have changed sources")
        for path,digest in old['output_sha256'].items():
            if sha(path)!=digest:raise ValueError("Existing pure-mechanism input artifact changed")
        return old
    output.mkdir(parents=True,exist_ok=True)
    with np.load(paths['effects'],allow_pickle=False) as z:
        effects=z['contributions'].astype(float);mask=z['module_mask'].astype(float)
        genes=z['genes'].astype(str);symbols=z['gene_symbols'].astype(str)
        std=z['train_gene_std'].astype(float);train_p=z['train_occupancy'].astype(float)
        if bool(z['synthetic'].item()) or bool(z['smoke_only'].item()):
            raise ValueError("Pure mechanism experiment requires the fitted real-data codec")
    axes,states,n_genes=effects.shape
    if (axes,states)!=(32,16) or genes.shape!=(n_genes,) or len(set(genes))!=n_genes:
        raise ValueError("Expected a 32-axis,16-state frozen codebook with unique gene IDs")
    if not np.isfinite(effects).all() or not np.allclose(train_p.sum(1),1):
        raise ValueError("Invalid frozen numeric effects or train occupancy")
    labels_doc=json.loads(paths['state_labels'].read_text())
    if labels_doc.get('effects_sha256')!=provenance['source_sha256']['effects'] or not labels_doc.get('meta',{}).get('complete'):
        raise ValueError("State labels must match the frozen dictionary")
    ann_doc=json.loads(paths['axis_annotations'].read_text())
    if ann_doc.get('status')!='complete':raise ValueError("Axis annotations are incomplete")
    annotations={int(v['axis_id']):v for v in ann_doc['axes']}
    if set(annotations)!=set(range(axes)):raise ValueError("Axis annotations must cover all axes")
    mechanisms=_mechanisms(json.loads(paths['mechanisms'].read_text()))
    case_doc=json.loads(paths['cases'].read_text())
    selected=sorted([v for v in case_doc['cases'] if v['status']=='ready' and float(v['dose_uM'])==5.],key=lambda v:v['case_id'])
    if not selected or any(v['drug'] not in mechanisms for v in selected):
        raise ValueError("Each selected case needs a primary-knowledge mechanism card")
    cards=make_axis_cards(effects,mask,genes,symbols,std,train_p,labels_doc['labels'],annotations)
    cards_path=output/'axis_cards.json'
    save_json(cards_path,{'axes':cards,'provenance':provenance,
        'label_scope':'Drug-agnostic frozen effects and language descriptions; no response-conditioned candidates'})
    with np.load(paths['baseline'],allow_pickle=False) as z:
        control_codes=z['codes'].copy();control_ids=z['cell_ids'].astype(str)
        control_x=z['observed_x'].astype(float);control_xr=z['reconstructed_x'].astype(float)
        if not np.array_equal(genes,z['genes'].astype(str)) or bool(z['synthetic'].item()):
            raise ValueError("Control vocabulary/provenance differs")
    lookup={v:i for i,v in enumerate(control_ids)}
    if len(lookup)!=len(control_ids):raise ValueError("Duplicate conditioning control cells")
    anon=anonymous_genes(n_genes);case_rows=[]
    for case in selected:
        requested=case['control_cell_ids']
        if len(requested)!=len(set(requested)) or len(requested)<20:
            raise ValueError("Ready case controls must be unique and meet minimum size")
        positions=np.asarray([lookup[v] for v in requested],dtype=int)
        public=public_id(case['case_id']);folder=output/public;folder.mkdir(exist_ok=True)
        p0=occupancy(control_codes[positions],states)
        request={'dose_uM':float(case['dose_uM']),'exposure_hours':float(case['exposure_time_hours']),
                 'cell_context':'A549 human lung adenocarcinoma cell line'}
        # No plate, replicate, vehicle, treated counts/IDs, or source filename in model context.
        context={'public_case_id':public,'request':request,'mechanism_card':mechanisms[case['drug']],
            'control_summary':_control_summary(control_x[positions],genes,symbols,anon),
            'provenance':provenance,
            'information_boundary':'Control+frozen code meanings+general drug mechanism only; no treatment responses or dose interpolation'}
        npz_path=folder/'case.npz';context_path=folder/'context.json'
        np.savez_compressed(npz_path,p0=p0,control_codes=control_codes[positions],
            control_mean=control_x[positions].mean(0),control_decoded_mean=control_xr[positions].mean(0),
            control_cell_ids=control_ids[positions],genes=genes,gene_symbols=symbols,anonymous_gene_ids=anon,synthetic=False)
        save_json(context_path,context)
        case_rows.append({'case_id':case['case_id'],'public_case_id':public,'opaque_id':public,'slug':public,
            'drug':case['drug'],'dose_uM':float(case['dose_uM']),'n_control':len(positions),
            'npz':str(npz_path),'context':str(context_path)})
    artifacts=[cards_path]+[Path(c[k]) for c in case_rows for k in ['npz','context']]
    manifest={'status':'complete','created_asia_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'provenance':provenance,'cases':case_rows,'cards_path':str(cards_path),'n_axes':axes,'n_states':states,'n_genes':n_genes,
        'condition_response_retrieval':False,'heldout_expression_accessed':False,
        'output_sha256':{str(p):sha(p) for p in artifacts},
        'private_metadata_warning':'Manifest case_id/drug are for orchestration and scoring only, never serialize whole manifest to a model',
        'development_status':'These5uM outcomes were already inspected; this is development, not a fresh independent test'}
    save_json(manifest_path,manifest)
    return manifest


def load_case_bundle(work,case_npz,case_context):
    work=Path(work).resolve();case_npz=Path(case_npz);case_context=Path(case_context)
    context=json.loads(case_context.read_text());prov=context['provenance']
    if sha(work/'lexa/exact_code_effects.npz')!=prov['source_sha256']['effects']:
        raise ValueError("Frozen effects changed")
    with np.load(case_npz,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(work/'lexa/exact_code_effects.npz',allow_pickle=False) as z:
        effects=z['contributions'].astype(float);genes=z['genes'].astype(str);symbols=z['gene_symbols'].astype(str)
        std=z['train_gene_std'].astype(float);train_p=z['train_occupancy'].astype(float)
    if not np.array_equal(genes,data['genes'].astype(str)) or not np.array_equal(data['anonymous_gene_ids'],anonymous_genes(len(genes))):
        raise ValueError("Case vocabulary or anonymous mapping differs")
    manifest=json.loads((case_npz.parent.parent/'manifest.json').read_text())
    for path in [case_npz,case_context]:
        if sha(path)!=manifest['output_sha256'][str(path.resolve())]:raise ValueError("Case artifact changed")
    private=next(c for c in manifest['cases'] if c['public_case_id']==context['public_case_id'])
    cards=json.loads(Path(manifest['cards_path']).read_text())['axes']
    if sha(manifest['cards_path'])!=manifest['output_sha256'][manifest['cards_path']]:raise ValueError("Cards changed")
    p0=data['p0'];baseline_effect=np.einsum('as,asg->ag',p0,effects)
    for axis,card in enumerate(cards):
        for state in card['states']:
            delta=effects[axis,state['state_id']]-baseline_effect[axis]
            state['relative_to_control_up']=gene_rows(delta,genes,symbols,data['anonymous_gene_ids'],std,1)
            state['relative_to_control_down']=gene_rows(delta,genes,symbols,data['anonymous_gene_ids'],std,-1)
    return {'case_id':private['case_id'],'public_case_id':context['public_case_id'],'opaque_id':context['public_case_id'],
        'request':context['request'],'mechanism_card':context['mechanism_card'],'control_summary':context['control_summary'],
        'p0':p0,'effects':effects,'genes':genes,'symbols':symbols,'std':std,'train_occupancy':train_p,
        'anonymous_gene_ids':data['anonymous_gene_ids'],'control_mean':data['control_mean'],
        'control_decoded_mean':data['control_decoded_mean'],'control_codes':data['control_codes'],
        'axis_cards':cards,'cards':{'axes':cards},
        'provenance':{**prov,'case_npz_sha256':sha(case_npz),'case_context_sha256':sha(case_context)}}


def _normalize_literal(text):
    """Punctuation/case normalization only: no synonyms or inferred pathways."""
    return " ".join(re.findall(r"[a-z0-9]+",text.lower()))


def _literal_sign(name, term):
    text=_normalize_literal(name);term=_normalize_literal(term)
    if not term:return None
    # Restrict direction to the same adjacent phrase, not an unrelated gene.
    middle=r"(?: (?:marker|markers|program|response|activity|associated)){0,3}"
    signs={}
    for direction,words in [('up','up|high|higher|increased|elevated|activated'),
                            ('down','down|low|lower|decreased|reduced|suppressed')]:
        signs[direction]=bool(re.search(r"\b"+re.escape(term)+middle+r" (?:"+words+r")\b",text) or
                             re.search(r"\b(?:"+words+r") "+re.escape(term)+r"\b",text))
    return next((k for k,v in signs.items() if v),None) if sum(signs.values())==1 else None


def literal_keyword_decisions(bundle):
    """Transparent, deliberately sparse keyword baseline, not response fitting.

    Exact normalized primary-mechanism program terms must have an adjacent
    explicit direction in a frozen state *name*. No gene synonym expansion,
    term-to-gene signatures, or inference from target inhibition is performed.
    Mixed/uncertain names or summaries abstain. At most three tied states are
    accepted; their requested weights are equal. Alpha is applied downstream.
    """
    terms=bundle['mechanism_card'].get('literal_program_terms',[])
    for term in terms:
        if term.get('direction') not in ('up','down') or not term.get('text'):
            raise ValueError('Literal terms require explicit up/down program activity')
    cards=bundle.get('axis_cards',bundle.get('cards',{}).get('axes',[]))
    if len(cards)!=len(bundle['p0']):raise ValueError('Literal baseline needs all axis cards')
    rows=[]
    for card in cards:
        axis=card['axis_id'];candidates=[]
        for state in card['states']:
            combined=_normalize_literal(state['name']+' '+state.get('summary',''))
            if re.search(r'\b(?:mixed|uncertain|unclear|inconsistent|not|no)\b',combined):continue
            matched=[];conflict=False
            for term in terms:
                sign=_literal_sign(state['name'],term['text'])
                if sign is not None:
                    if sign!=term['direction']:conflict=True
                    else:matched.append(term['text']+' '+sign)
            if matched and not conflict and bundle['train_occupancy'][axis,state['state_id']]>0:
                candidates.append((len(matched),state['state_id'],matched))
        best=max([x[0] for x in candidates],default=0)
        selected=[x for x in candidates if x[0]==best]
        if 0<len(selected)<=3:
            rows.append({'axis_id':axis,'decision':'shift',
                'targets':[[int(state),1.] for _,state,_ in selected],
                'evidence_refs':[f'A{axis}/C{state}' for _,state,_ in selected],
                'reason':'Literal state-name phrase match: '+', '.join(sorted({v for _,_,m in selected for v in m}))})
        else:
            rows.append({'axis_id':axis,'decision':'unknown','targets':[],'evidence_refs':[f'A{axis}'],
                'reason':'Abstain: no unambiguous explicit directional program phrase in state names.'})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parent/'runs/pilot_seen_dose')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--mechanisms',type=Path,required=True)
    args=parser.parse_args();output=args.output or args.work/'experiment_v4_pure_mechanism/inputs'
    m=prepare_inputs(args.work,output,args.mechanisms)
    print(json.dumps({'status':m['status'],'n_cases':len(m['cases']),'retrieval':False,'output':str(output)},ensure_ascii=False))


if __name__=='__main__':main()
