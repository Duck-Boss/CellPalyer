"""Isolation, semantics and artifact-integrity checks; no API or response data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pure_mechanism_data as data


class PureMechanismDataTest(unittest.TestCase):
    def fixture(self, root):
        work=root/'work';(work/'lexa').mkdir(parents=True)
        (work/'prepared').mkdir();(work/'experiment_v3_hierarchy_fixed').mkdir()
        genes=np.array([f'ENSG{i:06d}' for i in range(8)]);symbols=np.array([f'SYM{i}' for i in range(8)])
        effects=np.random.default_rng(8).normal(size=(32,16,8))
        np.savez(work/'lexa/exact_code_effects.npz',contributions=effects,module_mask=np.ones((32,8))/32,
                 genes=genes,gene_symbols=symbols,train_gene_std=np.ones(8),train_occupancy=np.ones((32,16))/16,
                 synthetic=False,smoke_only=False)
        labels={f'A{a}_C{s}':{'name':f'Expression state {s}','summary':'Signed gene contribution.'} for a in range(32) for s in range(16)}
        data.save_json(work/'lexa/labels.json',{'labels':labels,'meta':{'complete':True},'effects_sha256':data.sha(work/'lexa/exact_code_effects.npz')})
        data.save_json(work/'experiment_v3_hierarchy_fixed/axis_annotations.json',{'status':'complete',
            'axes':[{'axis_id':a,'name':'Gene expression program','functions':['gene expression'],'evidence_genes':genes[:2].tolist()} for a in range(32)]})
        ids=np.array([f'control_{i}' for i in range(24)]);codes=np.tile(np.arange(24)[:,None]%16,(1,32)).astype('int16')
        observed=np.random.default_rng(9).uniform(size=(24,8))
        decoded=effects[np.arange(32)[None,:],codes,:].sum(1)
        np.savez(work/'lexa/baseline.npz',codes=codes,cell_ids=ids,genes=genes,gene_symbols=symbols,
                 observed_x=observed,reconstructed_x=decoded,synthetic=False)
        case={'case_id':'drug=BMS|dose_uM=5|plate=PRIVATE_PLATE|replicate=PRIVATE_REP','drug':'BMS',
              'dose_uM':5.,'exposure_time_hours':24,'status':'ready','control_cell_ids':ids.tolist(),
              'treated_cell_ids':['SECRET_ANSWER_CELL'],'n_treated':999,'vehicle':'SECRET_VEHICLE'}
        data.save_json(work/'prepared/cases.json',{'cases':[case]})
        mechanism=root/'mechanisms.json';data.save_json(mechanism,{'drugs':{'BMS':{'drug_name':'BMS-345541',
            'mechanism':'Canonical NF-kappa-B inhibition.','literal_program_terms':[{'text':'NF-kappa-B','direction':'down'}]}}})
        return work,root/'inputs',mechanism

    def test_occupancy_and_invalid_codes(self):
        np.testing.assert_allclose(data.occupancy(np.array([[0,1],[1,1]]),3),[[.5,.5,0],[0,1,0]])
        for codes in [np.empty((0,2),int),np.array([[1.5]]),np.array([[3]]),np.array([[-1]])]:
            with self.assertRaises(ValueError):data.occupancy(codes,3)

    def test_opaque_gene_mapping(self):
        ids=data.anonymous_genes(2000)
        self.assertEqual(len(set(ids)),2000)
        self.assertTrue(np.array_equal(ids,data.anonymous_genes(2000)))
        self.assertFalse(np.array_equal(ids,np.array([f'G{i:04d}' for i in range(2000)])))
        self.assertRegex(data.public_id('drug=BMS|dose=5'),r'^P[0-9a-f]{12}$')
        self.assertNotEqual(data.public_id('drug=BMS'),data.public_id('drug=Dex'))

    def test_allowlisted_prepare_and_exact_relative_state_effects(self):
        with tempfile.TemporaryDirectory() as d:
            work,out,mechanism=self.fixture(Path(d));real_load=np.load;reads=[]
            def track(path,*args,**kwargs):
                reads.append(Path(path).name);return real_load(path,*args,**kwargs)
            with patch.object(data.np,'load',side_effect=track):manifest=data.prepare_inputs(work,out,mechanism)
            self.assertEqual(reads,['exact_code_effects.npz','baseline.npz'])
            self.assertEqual(len(manifest['cases']),1)
            row=manifest['cases'][0];context=Path(row['context']).read_text()
            for forbidden in ['SECRET_ANSWER_CELL','n_treated','treated_cell_ids','PRIVATE_PLATE','PRIVATE_REP','SECRET_VEHICLE','drug=BMS|']:
                self.assertNotIn(forbidden,context)
            bundle=data.load_case_bundle(work,row['npz'],row['context'])
            np.testing.assert_allclose(bundle['p0'].sum(1),1.)
            self.assertEqual(bundle['effects'].shape,(32,16,8))
            self.assertEqual(len(bundle['axis_cards']),32)
            self.assertTrue(all(len(c['states'])==16 for c in bundle['axis_cards']))
            axis=0;state=bundle['axis_cards'][axis]['states'][0]
            relative=bundle['effects'][axis,0]-bundle['p0'][axis]@bundle['effects'][axis]
            for sign,key in [(1,'relative_to_control_up'),(-1,'relative_to_control_down')]:
                for g in state[key]:
                    i=np.where(bundle['genes']==g['gene_id'])[0][0]
                    self.assertGreater(sign*g['delta'],0)
                    self.assertAlmostEqual(relative[i],g['delta'],places=5)
            # The additive decoded expectation agrees with the mean reconstruction.
            np.testing.assert_allclose(np.einsum('as,asg->g',bundle['p0'],bundle['effects']),bundle['control_decoded_mean'])
            self.assertEqual(data.prepare_inputs(work,out,mechanism),manifest)

    def test_resume_and_load_reject_tampered_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            work,out,mechanism=self.fixture(Path(d));manifest=data.prepare_inputs(work,out,mechanism)
            row=manifest['cases'][0];p=Path(row['context']);p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'artifact changed'):data.prepare_inputs(work,out,mechanism)
            with self.assertRaisesRegex(ValueError,'artifact changed'):data.load_case_bundle(work,row['npz'],row['context'])

    def test_labels_and_controls_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            work,out,mechanism=self.fixture(Path(d));p=work/'lexa/labels.json';doc=json.loads(p.read_text())
            doc['effects_sha256']='wrong';data.save_json(p,doc)
            with self.assertRaisesRegex(ValueError,'labels must match'):data.prepare_inputs(work,out,mechanism)
        with tempfile.TemporaryDirectory() as d:
            work,out,mechanism=self.fixture(Path(d));p=work/'prepared/cases.json';doc=json.loads(p.read_text())
            doc['cases'][0]['control_cell_ids'][0]=doc['cases'][0]['control_cell_ids'][1];data.save_json(p,doc)
            with self.assertRaisesRegex(ValueError,'must be unique'):data.prepare_inputs(work,out,mechanism)

    def test_drug_named_axis_annotation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            work,out,mechanism=self.fixture(Path(d));p=work/'experiment_v3_hierarchy_fixed/axis_annotations.json';doc=json.loads(p.read_text())
            doc['axes'][0]['name']='Nutlin response';data.save_json(p,doc)
            with self.assertRaisesRegex(ValueError,'drug identity'):data.prepare_inputs(work,out,mechanism)

    def test_literal_matching_abstains_without_explicit_direction(self):
        self.assertEqual(data._literal_sign('Strong p53-marker up state','p53'),'up')
        self.assertEqual(data._literal_sign('NF-kappa-B activity down','NF-kappa-B'),'down')
        self.assertIsNone(data._literal_sign('TP53I3 up','p53'))
        self.assertIsNone(data._literal_sign('p53 associated; unrelated gene down','p53'))
        states=[{'state_id':0,'name':'Strong p53-marker up state','summary':'Coordinated p53 markers.'},
                {'state_id':1,'name':'p53-marker up','summary':'Mixed directions.'},
                {'state_id':2,'name':'CDKN1A up','summary':'p53 pathway gene.'}]
        bundle={'p0':np.array([[1.,0,0],[1.,0,0]]),'train_occupancy':np.ones((2,3))/3,
                'mechanism_card':{'literal_program_terms':[{'text':'p53','direction':'up'}]},
                'axis_cards':[{'axis_id':0,'states':states},{'axis_id':1,'states':[states[2]]}]}
        rows=data.literal_keyword_decisions(bundle)
        self.assertEqual(rows[0]['targets'],[[0,1.]])
        self.assertEqual(rows[1]['decision'],'unknown')
        bundle['train_occupancy'][0,0]=0
        self.assertEqual(data.literal_keyword_decisions(bundle)[0]['decision'],'unknown')


if __name__=='__main__':unittest.main()
