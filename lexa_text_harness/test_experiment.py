import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiment import build_requests, feedback_statistics, metrics, run_methods, score


class ExperimentTests(unittest.TestCase):
    def make_project(self, temp, case_count=2):
        root = Path(temp)
        prepared, fitted, output = root / 'prepared', root / 'fitted', root / 'out'
        prepared.mkdir(); fitted.mkdir()
        genes = np.array(['G1', 'G2'])
        np.savez(fitted / 'exact_code_effects.npz', genes=genes,
                 contributions=np.array([[[0., 0.], [1., -1.]]], dtype=np.float32),
                 train_gene_std=np.ones(2), train_occupancy=np.ones((1, 2)) / 2)
        np.savez(fitted / 'baseline.npz', genes=genes, codes=np.zeros((2, 1), dtype=int),
                 cell_ids=np.array(['control1', 'control2']))
        cases = [{'case_id': f'case{i}', 'drug': 'Dex', 'dose_uM': 5., 'status': 'ready',
                  'control_cell_ids': ['control1', 'control2']} for i in range(case_count)]
        (prepared / 'cases.json').write_text(json.dumps({'cases': cases}))
        (prepared / 'manifest.json').write_text('{}')
        return prepared, fitted, output

    @staticmethod
    def fake_client(**kwargs):
        calls = [call['function']['name'] for message in kwargs['messages']
                 for call in message.get('tool_calls', [])]
        stages = [('get_context', {}), ('inspect_states', {'axis_id': 0}),
                  ('evaluate_edits', {'edits': [{'axis_id': 0, 'to_code': 1}]})]
        if len(calls) >= len(stages):
            message = {'role': 'assistant', 'content': '{"edits":[{"axis_id":0,"to_code":1}]}'}
        else:
            name, arguments = stages[len(calls)]
            message = {'role': 'assistant', 'content': None,
                       'tool_calls': [{'id': str(len(calls)), 'type': 'function',
                                       'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
        return {'choices': [{'message': message}]}

    def assert_targets_not_opened(self, prepared, output):
        actual_load = np.load
        def guarded_load(path, *args, **kwargs):
            if Path(path).name == 'evaluation.npz':
                self.fail('Held-out targets were opened before all predictions passed preflight')
            return actual_load(path, *args, **kwargs)
        with mock.patch('experiment.np.load', side_effect=guarded_load):
            with self.assertRaises(ValueError):
                score(prepared, output)

    def test_selection_can_expand_but_cannot_shrink_or_switch_source(self):
        with tempfile.TemporaryDirectory() as temp:
            prepared, fitted, output = self.make_project(temp)
            initial = build_requests(prepared, fitted, output / 'inputs', limit=1)
            expanded = build_requests(prepared, fitted, output / 'inputs', limit=0)
            self.assertEqual(len(initial), 1)
            self.assertEqual(len(expanded), 2)
            manifest = output / 'inputs/manifest.json'
            original = manifest.read_bytes()
            with self.assertRaisesRegex(ValueError, 'shrink'):
                build_requests(prepared, fitted, output / 'inputs', limit=1)
            self.assertEqual(manifest.read_bytes(), original)
            (prepared / 'manifest.json').write_text('{"different":true}')
            with self.assertRaisesRegex(ValueError, 'different prepared_manifest_sha256'):
                build_requests(prepared, fitted, output / 'inputs', limit=0)
            self.assertEqual(manifest.read_bytes(), original)

    def test_run_cannot_change_methods_before_overwriting_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            prepared, fitted, output = self.make_project(temp)
            run_methods(prepared, fitted, output, ['agent_no_labels'], self.fake_client, limit=1)
            original = (output / 'run_plan.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'methods'):
                run_methods(prepared, fitted, output, ['agent_no_labels', 'one_shot'], self.fake_client, limit=0)
            self.assertEqual((output / 'run_plan.json').read_bytes(), original)
            run_methods(prepared, fitted, output, ['agent_no_labels'], self.fake_client, limit=0)
            self.assertEqual(len(json.loads((output / 'inputs/manifest.json').read_text())['cases']), 2)

    def test_missing_later_case_prevents_opening_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            prepared, fitted, output = self.make_project(temp)
            paths = run_methods(prepared, fitted, output, ['agent_no_labels'], self.fake_client, limit=0)
            Path(paths[-1]).unlink()
            self.assert_targets_not_opened(prepared, output)

    def test_prediction_checksum_or_failed_evaluation_prevents_opening_targets(self):
        for problem in ['checksum', 'status', 'provenance']:
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as temp:
                prepared, fitted, output = self.make_project(temp)
                paths = run_methods(prepared, fitted, output, ['agent_no_labels'], self.fake_client, limit=0)
                path = Path(paths[-1])
                result = json.loads(path.read_text())
                if problem == 'checksum':
                    with Path(result['evaluation']['artifact']).open('ab') as handle:
                        handle.write(b'changed')
                elif problem == 'status':
                    result['status'] = 'transport_error'
                else:
                    result['evaluation']['provenance']['baseline_sha256'] = 'different'
                path.write_text(json.dumps(result))
                self.assert_targets_not_opened(prepared, output)

    def test_feedback_statistics_separate_same_round_candidates(self):
        trace = {'events': [
            {'round': 3, 'tool': 'evaluate_edits', 'success': True},
            {'round': 3, 'tool': 'evaluate_edits', 'success': True},
            {'round': 4, 'tool': 'evaluate_edits', 'success': False},
            {'round': 5, 'tool': 'evaluate_edits', 'success': True}]}
        self.assertEqual(feedback_statistics(trace), {
            'successful_evaluation_api_rounds': [3, 3, 5],
            'evaluations_after_prior_round_feedback': 1,
            'distinct_evaluation_api_rounds': 2, 'same_round_extra_evaluations': 1})

    def test_delta_metrics_do_not_reward_absolute_baseline(self):
        observed = np.array([1., -2., 0.5, 0.])
        unchanged = metrics(np.zeros(4), observed)
        perfect = metrics(observed, observed)
        self.assertIsNone(unchanged['delta_pearson'])
        self.assertGreater(unchanged['delta_rmse'], 0)
        self.assertEqual(unchanged['top100_direction_accuracy'], 0)
        self.assertAlmostEqual(perfect['delta_pearson'], 1)
        self.assertEqual(perfect['delta_rmse'], 0)

    def test_real_workspace_loop_runs_without_evaluation_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared, fitted, output = root / 'prepared', root / 'fitted', root / 'out'
            prepared.mkdir(); fitted.mkdir()
            genes = np.array(['G1', 'G2', 'G3', 'G4'])
            effect = np.zeros((2, 3, 4), np.float32)
            effect[0, 1] = [1, -2, 0.5, 0]
            np.savez(fitted / 'exact_code_effects.npz', genes=genes, contributions=effect,
                     train_gene_std=np.ones(4), train_occupancy=np.ones((2, 3)) / 3)
            np.savez(fitted / 'baseline.npz', genes=genes, codes=np.zeros((2, 2), dtype=int),
                     cell_ids=np.array(['control1', 'control2']))
            case = {'case_id': 'case1', 'drug': 'Dex', 'dose_uM': 1., 'status': 'ready',
                    'control_cell_ids': ['control1', 'control2'], 'n_treated': 2}
            (prepared / 'cases.json').write_text(json.dumps({'cases': [case]}))
            (prepared / 'manifest.json').write_text('{}')
            counter = iter([
                {'role': 'assistant', 'content': None, 'tool_calls': [{'id': '1', 'type': 'function', 'function': {'name': 'get_context', 'arguments': '{}'}}]},
                {'role': 'assistant', 'content': None, 'tool_calls': [{'id': '2', 'type': 'function', 'function': {'name': 'inspect_states', 'arguments': '{"axis_id":0}'}}]},
                {'role': 'assistant', 'content': None, 'tool_calls': [{'id': '3', 'type': 'function', 'function': {'name': 'evaluate_edits', 'arguments': '{"edits":[{"axis_id":0,"to_code":1}]}'}}]},
                {'role': 'assistant', 'content': '{"edits":[{"axis_id":0,"to_code":1}],"summary":"hypothesis","limitations":[]}'}])
            def fake(**kwargs):
                serialized = json.dumps(kwargs)
                self.assertNotIn('x_test', serialized)
                self.assertNotIn('treated_cell_ids', serialized)
                return {'choices': [{'message': next(counter)}]}
            paths = run_methods(prepared, fitted, output, ['agent_no_labels'], fake, limit=4)
            self.assertEqual(len(paths), 1)
            result = json.loads(Path(paths[0]).read_text())
            self.assertTrue(result['decoder_evaluated'])
            self.assertFalse((prepared / 'evaluation.npz').exists())
            self.assertEqual(result['mode'], 'perturbation_prediction')
            # Only now create observed target data, proving it was unavailable to the agent.
            np.savez(prepared / 'evaluation.npz', genes=genes, x_test=np.tile(effect[0, 1], (2, 1)),
                     test_case_ids=np.array(['case1', 'case1']), control_x=np.zeros((2, 4)),
                     control_cell_ids=np.array(['control1', 'control2']))
            np.savez(prepared / 'prepared.npz', x_train=np.zeros((2, 4)), train_cell_ids=np.array(['control1', 'control2']))
            (prepared / 'metadata.csv').write_text('cell_id,drug,dose_uM,plate,replicate\ncontrol1,Dex,0,A,1\ncontrol2,Dex,0,A,1\n')
            rows = score(prepared, output)
            by_method = {row['method']: row for row in rows}
            self.assertEqual(by_method['agent_no_labels']['delta_rmse'], 0)
            self.assertGreater(by_method['no_edit']['delta_rmse'], 0)
            # A failed stored result must be retried in a new attempt, never skipped.
            result['status'] = 'transport_error'
            Path(paths[0]).write_text(json.dumps(result))
            with self.assertRaises(RuntimeError):
                run_methods(prepared, fitted, output, ['agent_no_labels'], fake, limit=4)
            attempts = list((Path(paths[0]).parent / 'attempts').iterdir())
            self.assertEqual(len(attempts), 2)
            with self.assertRaises(ValueError):
                score(prepared, output)


if __name__ == '__main__':
    unittest.main()
