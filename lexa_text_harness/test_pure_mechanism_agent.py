"""Pure text prediction protocol, leakage boundaries and exact arithmetic."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import pure_mechanism_agent as agent


def inputs(axes=32):
    rng=np.random.default_rng(529)
    effects=rng.normal(size=(axes,16,6))
    p0=np.zeros((axes,16));p0[:,0]=1
    genes=np.array([f'ENSG_SECRET_{i}' for i in range(6)])
    symbols=np.array([f'GENE_SECRET_{i}' for i in range(6)])
    bundle={'case_id':'DrugSecret_DMSO_5uM_24h_PRIVATE','opaque_id':'case_e98a17',
        'request':{'dose_uM':5.,'exposure_hours':24.,'cell_context':'A549 human cells',
                   'drug':'DrugSecret','vehicle':'DMSO','condition_id':'PRIVATE_CONDITION'},
        'mechanism_card':{'drug':'DrugSecret','target':'TARGET_SECRET',
                          'sources':['https://mechanism-secret.invalid/DrugSecret']},
        'p0':p0,'effects':effects,'genes':genes,'symbols':symbols,'std':np.ones(6),
        'anonymous_gene_ids':np.array(['G0004','G0001','G0005','G0002','G0000','G0003']),
        'train_occupancy':np.full((axes,16),1/16),'control_mean':np.arange(6)+2.,
        'control_decoded_mean':np.arange(6)+2.,'control_codes':np.zeros((7,axes),dtype=int),
        'control_summary':{'unsafe_gene_list':['BAD_CONTROL_SENTINEL']},
        'reference_observed_delta':'FORBIDDEN_RESPONSE','training_library':'FORBIDDEN_LIBRARY',
        'provenance':{'test':True}}
    cards=[{'axis_id':a,'scope':'SCOPE_SECRET_GENE_SECRET_0',
        'language_annotation':{'name':'ANNOTATION_SECRET','summary':'FUNCTION_SECRET'},
        'states':[{'state_id':s,'name':f'STATE_SECRET_{s}',
                   'summary':'GENE_SECRET_0 up / GENE_SECRET_1 down',
                   'positive_genes':[{'gene_id':'BAD_UNUSED_GENE'}]} for s in range(16)]}
        for a in range(axes)]
    return bundle,cards


def plan(visible=True):
    return {'hypotheses':[{'id':'H1','program':'possible transcriptional program',
        'direction':'increase' if visible else 'uncertain','basis':'secondary' if visible else 'unknown',
        'conditions':'context dependent','gene_expectations':[{'gene':'GENE_SECRET_0','direction':'increase'}] if visible else [],
        'evidence_refs':['M' if visible else 'C'],'reason':'Supplied context only'}],
        'summary':'Hypotheses before code inspection','limitations':['No measured treatment response']}


def row(axis,state=1,decision='shift'):
    return {'axis_id':axis,'decision':decision,'targets':[[state,1.]] if decision=='shift' else [],
        'evidence_refs':['H1',f'A{axis}/C{state}'],'reason':'Direction supported by supplied hypothesis'}


def response(axes,partial=False):
    return {'updates' if partial else 'decisions':[row(a) for a in axes],
            'summary':'State preferences','limitations':['Direction and amplitude remain uncertain']}


class Client:
    model='unit-test-model'
    url='https://unit-test.invalid/chat/completions'
    _disable_thinking=True

    def __init__(self,invalid_first=False,always_invalid=False,joint_change=True):
        self.calls=0;self.invalid_first=invalid_first;self.always_invalid=always_invalid
        self.joint_change=joint_change;self.requests=[]

    def __call__(self,*,messages,tools):
        self.calls+=1;self.requests.append(copy.deepcopy(messages))
        payload=json.loads(messages[1]['content'])
        if self.always_invalid or (self.invalid_first and self.calls==1):
            content='{"invalid":true}'
        elif payload['stage']=='mechanism_hypotheses':
            content=json.dumps(plan('mechanism_card' in payload['context']))
        elif payload['stage']=='local_state_preferences':
            result=response(payload['requested_axes'])
            for r in result['decisions']:
                if r['axis_id']%3==1:r.update(decision='keep',targets=[])
                if r['axis_id']%3==2:r.update(decision='unknown',targets=[])
            content=json.dumps(result)
        else:
            result=response([],True)
            if self.joint_change:result['updates']=[row(0,2)]
            content=json.dumps(result)
        return {'choices':[{'message':{'role':'assistant','content':content}}],
                'usage':{'prompt_tokens':10,'completion_tokens':10}}


class ProtocolTests(unittest.TestCase):
    def test_exact_mechanism_source_refs_visible_only(self):
        value=plan();value['hypotheses'][0]['evidence_refs']=['M1','M2']
        self.assertEqual(agent.parse_plan(json.dumps(value),True,['M1','M2']),value)
        for refs in (None,['M1'],['M3']):
            with self.assertRaises(agent.ValidationError):agent.parse_plan(json.dumps(value),True,refs)
        with self.assertRaises(agent.ValidationError):agent.parse_plan(json.dumps(value),False,['M1','M2'])
        value['hypotheses'][0]['evidence_refs']=['C']
        self.assertEqual(agent.parse_plan(json.dumps(value),False,['M1','M2']),value)

    def test_valid_plan_and_partial(self):
        self.assertEqual(agent.parse_plan('```json\n'+json.dumps(plan())+'\n```'),plan())
        self.assertEqual(agent.parse_decisions(json.dumps(response([],True)),[0,1],16,plan(),True)['updates'],[])
        with self.assertRaises(agent.ValidationError):agent.parse_plan(json.dumps(plan()),False)

    def test_bad_decisions(self):
        cases=[]
        for mutator in [
            lambda x:x['decisions'].pop(),
            lambda x:x['decisions'][1].update(axis_id=0),
            lambda x:x['decisions'][0].update(axis_id=False),
            lambda x:x['decisions'][0].update(targets=[[16,1]]),
            lambda x:x['decisions'][0].update(targets=[[False,1]]),
            lambda x:x['decisions'][0].update(targets=[[1,0]]),
            lambda x:x['decisions'][0].update(targets=[[1,float('nan')]]),
            lambda x:x['decisions'][0].update(targets=[[1,True]]),
            lambda x:x['decisions'][0].update(targets=[[1,1],[1,2]]),
            lambda x:x['decisions'][0].update(decision='keep'),
            lambda x:x['decisions'][0].update(evidence_refs=['H9']),
            lambda x:x['decisions'][0].update(evidence_refs=['A1/C1']),
            lambda x:x['decisions'][0].update(decision='invent')]:
            value=response([0,1]);mutator(value);cases.append(value)
        for value in cases:
            with self.subTest(value=value),self.assertRaises(agent.ValidationError):
                agent.parse_decisions(json.dumps(value),[0,1],16,plan())
        occupancy=np.ones((2,16));occupancy[0,1]=0
        with self.assertRaises(agent.ValidationError):
            agent.parse_decisions(json.dumps(response([0,1])),[0,1],16,plan(),train_occupancy=occupancy)


class NumericalTests(unittest.TestCase):
    def test_exact_decoding_and_unknown(self):
        b,_=inputs(3);decisions=[row(0),row(1,decision='keep'),row(2,decision='unknown')]
        decisions[0]['targets']=[[1,1],[2,3]]
        e=agent.evaluate_decisions(b,decisions,.3,plan())
        expected=.3*(.25*b['effects'][0,1]+.75*b['effects'][0,2]-b['effects'][0,0])
        np.testing.assert_allclose(e['delta_mean'],expected,atol=1e-14)
        np.testing.assert_allclose(e['q'].sum(1),1)
        np.testing.assert_array_equal(e['q'][1:],b['p0'][1:])
        self.assertEqual(e['feedback']['decision_counts'],{'shift':1,'keep':1,'unknown':1})
        self.assertEqual(e['q'][0,1],.075)
        for a in (.1,.6):
            np.testing.assert_allclose(agent.evaluate_decisions(b,decisions,a)['delta_mean'],expected*a/.3,atol=1e-14)
        for invalid in (float('nan'),float('inf'),True,-1,2):
            with self.assertRaises(ValueError):agent.evaluate_decisions(b,decisions,invalid)

    def test_feedback_is_model_consistency_only(self):
        b,_=inputs(2);rows=[row(0),row(1)]
        full=agent.evaluate_decisions(b,rows,.3,plan())['feedback']
        self.assertEqual(full['hypothesis_gene_checks'][0]['gene'],'GENE_SECRET_0')
        hidden=agent.evaluate_decisions(b,rows,.3,plan(),'no_semantics')['feedback']
        text=json.dumps(hidden)
        self.assertNotIn('GENE_SECRET',text);self.assertNotIn('ENSG_SECRET',text)
        self.assertIsInstance(hidden['hypothesis_gene_checks'],str)


class PayloadTests(unittest.TestCase):
    def test_source_refs_are_explicit_and_hidden_when_ablated(self):
        b,_=inputs()
        b['mechanism_card']['sources']=[{'id':'M1','title':'source one'}, {'id':'M2','title':'source two'}]
        for arm in agent.ARMS:
            payload=agent.build_plan_payload(b,arm)
            expected=['C','M','M1','M2'] if arm in ('full','no_semantics') else ['C']
            self.assertEqual(payload['available_evidence_refs'],expected)
            if len(expected)==1:
                self.assertNotIn('M1',json.dumps(payload));self.assertNotIn('M2',json.dumps(payload))

    def test_four_arm_information_boundary(self):
        b,c=inputs()
        for arm in agent.ARMS:
            p=plan(arm in ('full','no_semantics'))
            start=agent.build_plan_payload(b,arm)
            self.assertNotIn('A0/C',json.dumps(start))
            local=agent.build_local_payload(b,c,p,list(range(8)),arm)
            rows=response(range(32))['decisions']
            feedback=agent.evaluate_decisions(b,rows,.3,p,arm)['feedback']
            joint=agent.build_joint_payload(b,c,p,{'decisions':rows,'feedback':feedback},arm)
            text=json.dumps([start,local,joint])
            for forbidden in ('PRIVATE_CONDITION','DMSO','DrugSecret_DMSO','FORBIDDEN_RESPONSE','FORBIDDEN_LIBRARY','BAD_CONTROL_SENTINEL','BAD_UNUSED_GENE'):
                self.assertNotIn(forbidden,text)
            if arm in ('no_mechanism','neither'):
                for forbidden in ('DrugSecret','TARGET_SECRET','mechanism-secret.invalid'):
                    self.assertNotIn(forbidden,text)
            if arm in ('no_semantics','neither'):
                for payload in (local,joint):
                    evidence=copy.deepcopy(payload);evidence.pop('frozen_hypotheses');evidence['context'].pop('mechanism_card',None)
                    encoded=json.dumps(evidence)
                    for forbidden in ('ENSG_SECRET','GENE_SECRET','SCOPE_SECRET','ANNOTATION_SECRET','FUNCTION_SECRET','STATE_SECRET','BAD_UNUSED_GENE'):
                        self.assertNotIn(forbidden,encoded)
                self.assertIn('G000',text)
            else:self.assertIn('GENE_SECRET',text)

    def test_numeric_evidence_equal_across_semantic_ablation(self):
        b,c=inputs(2)
        full=agent.build_plan_payload(b,'full')['context']['control_expressed_features']
        anon=agent.build_plan_payload(b,'no_semantics')['context']['control_expressed_features']
        self.assertEqual([r['mean'] for r in full],[r['mean'] for r in anon])
        self.assertEqual([r['feature'] for r in anon],list(b['anonymous_gene_ids'][::-1]))
        a=agent.build_local_payload(b,c,plan(),[0,1],'full')['axes']
        z=agent.build_local_payload(b,c,plan(),[0,1],'no_semantics')['axes']
        for real,hidden in zip(a,z):
            self.assertEqual(real['control_occupancy'],hidden['control_occupancy'])
            for r,h in zip(real['states'],hidden['states']):
                self.assertEqual(r['train_occupancy'],h['train_occupancy'])
                for key in ('effect_from_control','effect_from_axis_reference'):
                    for direction in ('up','down'):
                        self.assertEqual([(g['delta'],g['scaled_delta']) for g in r[key][direction]],
                                         [(g['delta'],g['scaled_delta']) for g in h[key][direction]])


class RunnerTests(unittest.TestCase):
    def test_runner_accepts_exact_supplied_source_refs(self):
        class SourceClient(Client):
            def __call__(self,**kwargs):
                result=super().__call__(**kwargs)
                payload=json.loads(kwargs['messages'][1]['content'])
                if payload['stage']=='mechanism_hypotheses':
                    value=json.loads(result['choices'][0]['message']['content'])
                    value['hypotheses'][0]['evidence_refs']=['M1','M2']
                    result['choices'][0]['message']['content']=json.dumps(value)
                return result
        b,c=inputs(2);b['mechanism_card']['sources']=[{'id':'M1'},{'id':'M2'}]
        with tempfile.TemporaryDirectory() as directory:
            client=SourceClient();result=agent.run_case(b,c,client,directory)
            self.assertEqual(client.calls,3)
            self.assertEqual(result['complete']['api_attempts'],3)

    def test_six_calls_joint_patch_three_strengths_and_cache(self):
        b,c=inputs();client=Client()
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            result=agent.run_case(b,c,client,out)
            self.assertEqual(client.calls,6);self.assertEqual(result['complete']['api_attempts'],6)
            self.assertEqual(result['local']['decisions'][0]['targets'],[[1,1.]])
            self.assertEqual(result['joint']['decisions'][0]['targets'],[[2,1.]])
            for name in ('local','joint'):
                with np.load(out/f'{name}.npz') as z:
                    self.assertEqual(z['q_by_alpha'].shape,(3,32,16))
                    np.testing.assert_array_equal(z['q'],z['q_by_alpha'][1])
                    np.testing.assert_allclose(z['delta_by_alpha'][0]*3,z['delta_mean'],atol=1e-13)
                    np.testing.assert_allclose(z['delta_by_alpha'][2]/2,z['delta_mean'],atol=1e-13)
                    np.testing.assert_allclose(z['delta_mean'],np.einsum('as,asg->g',z['q']-z['p0'],b['effects']),atol=1e-13)
            cached=Client();agent.run_case(b,c,cached,out)
            self.assertEqual(cached.calls,0)
            changed=copy.deepcopy(b);changed['effects'][0,0,0]+=.1
            with self.assertRaisesRegex(ValueError,'changed'):agent.run_case(changed,c,cached,out)
            changed_client=Client();changed_client.model='different-model'
            with self.assertRaisesRegex(ValueError,'changed'):agent.run_case(b,c,changed_client,out)

    def test_repairs_are_bounded_and_cached(self):
        b,c=inputs();client=Client(invalid_first=True)
        with tempfile.TemporaryDirectory() as directory:
            result=agent.run_case(b,c,client,directory)
            self.assertEqual(client.calls,7);self.assertEqual(result['complete']['api_attempts'],7)
            cached=Client();agent.run_case(b,c,cached,directory);self.assertEqual(cached.calls,0)
        with tempfile.TemporaryDirectory() as directory:
            invalid=Client(always_invalid=True)
            with self.assertRaises(agent.StageFailure):agent.run_case(b,c,invalid,directory)
            self.assertEqual(invalid.calls,3)
            cached=Client()
            with self.assertRaises(agent.StageFailure):agent.run_case(b,c,cached,directory)
            self.assertEqual(cached.calls,0)

    def test_unknown_request_is_counted_not_reissued(self):
        b,_=inputs();payload=agent.build_plan_payload(b,'full');calls=[]
        def interrupted(**kwargs):calls.append(kwargs);raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                agent._run_stage(folder,payload,agent.PLAN_PROMPT,agent.parse_plan,interrupted,'unit-fingerprint',2)
            client=Client()
            stage=agent._run_stage(folder,payload,agent.PLAN_PROMPT,agent.parse_plan,client,'unit-fingerprint',2)
            self.assertEqual(len(calls),1);self.assertEqual(client.calls,1);self.assertEqual(stage['attempts_used'],2)
            first=json.loads((folder/'attempt_01.status.json').read_text())
            self.assertIn('unknown outcome',first['error'])

    def test_tampered_response_is_rejected(self):
        b,c=inputs(2)
        with tempfile.TemporaryDirectory() as directory:
            agent.run_case(b,c,Client(),directory)
            path=Path(directory)/'requests/plan/attempt_01.response.json'
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'response changed'):
                agent.run_case(b,c,Client(),directory)


if __name__=='__main__':unittest.main()
