"""Offline agent tests: no API calls, credentials, or real biological claims."""

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from agent import ChatCompletionsClient, TransportError, run_agent


def response(*calls, content=None):
    tool_calls = [
        {"id": f"call_{index}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        for index, (name, arguments) in enumerate(calls)
    ]
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}]}


class FakeWorkspace:
    def __init__(self):
        self.evaluated = []

    def get_context(self):
        return {"num_axes": 2, "codebook_size": 3, "synthetic": True, "support": "unknown"}

    def inspect_states(self, axis_id):
        if not 0 <= axis_id < 2:
            raise ValueError("Axis out of range.")
        return {"axis_id": axis_id, "cards": [{"code_id": 0}, {"code_id": 1}, {"code_id": 2}]}

    def evaluate_edits(self, edits):
        if any(edit["to_code"] >= 3 for edit in edits):
            raise ValueError("Code out of range.")
        self.evaluated.append(edits)
        return {"synthetic": True, "delta_mean": [float(edits[0]["to_code"]), -1.0], "support": "unknown"}


class ScriptedTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = 0
        self.messages = []

    def __call__(self, *, messages, tools):
        self.calls += 1
        self.messages.append(messages)
        return self.responses.pop(0)


class AgentTests(unittest.TestCase):
    def test_normal_loop_and_revision_uses_last_actual_evaluation(self):
        workspace = FakeWorkspace()
        transport = ScriptedTransport(
            response(("get_context", {}), ("inspect_states", {"axis_id": 0})),
            response(("evaluate_edits", {"edits": [{"axis_id": 0, "to_code": 1}]})),
            response(("evaluate_edits", {"edits": [{"axis_id": 0, "to_code": 2}]})),
            response(content=json.dumps({"edits": [{"axis_id": 0, "to_code": 2}], "summary": "A numerical edit."})),
        )
        result = run_agent(workspace, transport)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["decoder_evaluated"])
        self.assertEqual(result["last_evaluated_proposal"]["edits"], [{"axis_id": 0, "to_code": 2}])
        self.assertTrue(result["model_final_matches_last_evaluation"])
        self.assertFalse(result["biological_effect_validated"])
        self.assertEqual(len(workspace.evaluated), 2)

    def test_bad_tool_arguments_can_recover(self):
        workspace = FakeWorkspace()
        transport = ScriptedTransport(
            response(("get_context", {}), ("inspect_states", {"axis_id": "0"})),
            response(("inspect_states", {"axis_id": 0})),
            response(("evaluate_edits", {"edits": [{"axis_id": 0, "to_code": 1}]})),
            response(content=json.dumps({"edits": [{"axis_id": 0, "to_code": 1}]})),
        )
        result = run_agent(workspace, transport)
        self.assertTrue(result["decoder_evaluated"])
        tool_errors = [json.loads(message["content"]) for message in result["trace"]["messages"] if message["role"] == "tool"]
        self.assertTrue(any(item.get("error") == "invalid_tool_request" for item in tool_errors))

    def test_duplicate_axis_has_specific_feedback_and_can_recover(self):
        workspace = FakeWorkspace()
        duplicate = [{'axis_id': 0, 'from_code': 0, 'to_code': 1},
                     {'axis_id': 0, 'from_code': 2, 'to_code': 1}]
        transport = ScriptedTransport(
            response(('get_context', {}), ('inspect_states', {'axis_id': 0})),
            response(('evaluate_edits', {'edits': duplicate})),
            response(('evaluate_edits', {'edits': [duplicate[0]]})),
            response(content=json.dumps({'edits': [duplicate[0]]})),
        )
        result = run_agent(workspace, transport)
        errors = [json.loads(m['content']) for m in result['trace']['messages']
                  if m['role'] == 'tool' and 'duplicate_axis' in m['content']]
        self.assertEqual(len(errors), 1)
        self.assertIn('axis_id 0', errors[0]['message'])
        self.assertIn('Choose one', errors[0]['message'])
        self.assertFalse(errors[0]['edit_applied'])
        self.assertEqual(workspace.evaluated, [[duplicate[0]]])
        self.assertTrue(result['decoder_evaluated'])

    def test_early_final_is_explicitly_unevaluated(self):
        workspace = FakeWorkspace()
        transport = ScriptedTransport(response(content=json.dumps({"edits": [{"axis_id": 0, "to_code": 1}], "summary": "This drug definitely works."})))
        result = run_agent(workspace, transport)
        self.assertFalse(result["decoder_evaluated"])
        self.assertIsNone(result["last_evaluated_proposal"])
        self.assertFalse(result["biological_effect_validated"])
        self.assertEqual(workspace.evaluated, [])

    def test_budget_stops_without_extra_request(self):
        transport = ScriptedTransport(response(("get_context", {})), response(("inspect_states", {"axis_id": 0})))
        result = run_agent(FakeWorkspace(), transport, max_rounds=2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(transport.calls, 2)
        self.assertFalse(result["decoder_evaluated"])

    def test_final_cannot_replace_evaluated_proposal(self):
        transport = ScriptedTransport(
            response(("get_context", {}), ("inspect_states", {"axis_id": 0})),
            response(("evaluate_edits", {"edits": [{"axis_id": 0, "to_code": 1}]})),
            response(content=json.dumps({"edits": [{"axis_id": 0, "to_code": 2}]})),
        )
        result = run_agent(FakeWorkspace(), transport)
        self.assertEqual(result["last_evaluated_proposal"]["edits"], [{"axis_id": 0, "to_code": 1}])
        self.assertFalse(result["model_final_matches_last_evaluation"])

    def test_inspection_required_and_boolean_ids_rejected(self):
        transport = ScriptedTransport(
            response(("get_context", {}), ("evaluate_edits", {"edits": [{"axis_id": 0, "to_code": 1}]})),
            response(("inspect_states", {"axis_id": True})),
        )
        result = run_agent(FakeWorkspace(), transport, max_rounds=2)
        self.assertFalse(result["decoder_evaluated"])
        self.assertEqual(sum(not event["success"] for event in result["trace"]["events"]), 2)

    def test_transport_exception_details_do_not_enter_trace(self):
        def broken(**_):
            raise RuntimeError("secret-api-key should never be logged")

        result = run_agent(FakeWorkspace(), broken)
        self.assertEqual(result["status"], "transport_error")
        self.assertNotIn("secret-api-key", json.dumps(result))

    def test_fraction_and_source_survive_evaluation(self):
        edits = [{"axis_id": 0, "to_code": 2, "from_code": 1, "fraction": 0.25}]
        transport = ScriptedTransport(
            response(("get_context", {}), ("inspect_states", {"axis_id": 0})),
            response(("evaluate_edits", {"edits": edits})),
            response(content=json.dumps({"edits": edits})),
        )
        workspace = FakeWorkspace()
        result = run_agent(workspace, transport)
        self.assertEqual(workspace.evaluated, [edits])
        self.assertTrue(result["model_final_matches_last_evaluation"])

    def test_invalid_fraction_is_rejected(self):
        for fraction in (-0.1, 1.1, True, float("nan")):
            with self.subTest(fraction=fraction):
                transport = ScriptedTransport(
                    response(("get_context", {}), ("inspect_states", {"axis_id": 0})),
                    response(("evaluate_edits", {"edits": [{"axis_id": 0, "to_code": 1, "fraction": fraction}]})),
                )
                workspace = FakeWorkspace()
                result = run_agent(workspace, transport, max_rounds=2)
                self.assertFalse(result["decoder_evaluated"])
                self.assertEqual(workspace.evaluated, [])


class TransportTests(unittest.TestCase):
    def test_custom_endpoint_has_no_deepseek_extension(self):
        sent = []

        def fake_urlopen(request, timeout):
            sent.append((request, timeout))
            return io.BytesIO(json.dumps(response(content="{}")).encode())

        with patch("agent.urllib.request.urlopen", fake_urlopen):
            client = ChatCompletionsClient("explicit-model", "test-secret", "https://example.test/v1")
            client(messages=[], tools=[])
        payload = json.loads(sent[0][0].data)
        self.assertNotIn("thinking", payload)
        self.assertEqual(sent[0][0].full_url, "https://example.test/v1/chat/completions")
        self.assertLessEqual(sent[0][1], 45)

    def test_official_endpoint_disables_thinking(self):
        sent = []

        def fake_urlopen(request, timeout):
            sent.append(json.loads(request.data))
            return io.BytesIO(json.dumps(response(content="{}")).encode())

        with patch("agent.urllib.request.urlopen", fake_urlopen):
            ChatCompletionsClient("explicit-model", "test-secret")(messages=[], tools=[])
        self.assertEqual(sent[0]["thinking"], {"type": "disabled"})

    def test_http_error_body_is_never_exposed(self):
        error = urllib.error.HTTPError("https://example.test", 401, "secret-reason", {}, io.BytesIO(b"secret-body-and-key"))
        with patch("agent.urllib.request.urlopen", side_effect=error):
            with self.assertRaises(TransportError) as caught:
                ChatCompletionsClient("explicit-model", "test-secret")(messages=[], tools=[])
        self.assertIn("401", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
