"""API contract checks that never load an embedding model or call a provider."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend_files"))
spec = importlib.util.spec_from_file_location("aurora_backend_app", ROOT / "backend_files/app.py")
backend_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend_app)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.store = object()
        self.app = backend_app.create_app(vectorstore=self.store)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_invalid_requests_never_reach_generation(self):
        bad_bodies = [None, [], {}, {"query": " "}, {"query": 5},
                      {"query": "Question?", "k": True}, {"query": "Question?", "k": 0},
                      {"query": "Question?", "temperature": float("nan")},
                      {"query": "Question?", "top_p": 0}, {"query": "Question?", "max_tokens": -1}]
        with patch.object(backend_app, "rag") as generate:
            for endpoint in ("/v1/answer", "/v1/answer_with_relevant_chunks"):
                for body in bad_bodies:
                    with self.subTest(endpoint=endpoint, body=body):
                        response = self.client.post(endpoint, json=body)
                        self.assertEqual(response.status_code, 400)
                        self.assertIn("error", response.json)
            generate.assert_not_called()

    def test_answer_only_keeps_the_frontend_response_contract_and_defaults(self):
        with patch.object(backend_app, "rag", return_value=("Grounded answer", ["Evidence"])) as pipeline:
            response = self.client.post("/v1/answer", json={"query": "A hotel question?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"answer": "Grounded answer"})
        self.assertEqual(pipeline.call_args.kwargs, {
            "query": "A hotel question?", "k": 4, "model_name": "gpt-4o-mini",
            "temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "vectorstore": self.store,
        })

    def test_nondefault_request_parameters_are_forwarded(self):
        parameters = {"query": "A hotel question?", "k": 2, "model_name": "gpt-4o-mini",
                      "temperature": 0.17, "top_p": 0.93, "max_tokens": 321}
        with patch.object(backend_app, "rag", return_value=("Answer", ["Evidence"])) as pipeline:
            response = self.client.post("/v1/answer_with_relevant_chunks", json=parameters)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(pipeline.call_args.kwargs, {**parameters, "vectorstore": self.store})
        self.assertEqual(response.json, {**parameters, "answer": "Answer", "relevant_chunks": ["Evidence"]})

    def test_upstream_error_details_are_not_returned(self):
        with patch.object(backend_app, "rag", side_effect=RuntimeError("secret-provider-request-detail")):
            response = self.client.post("/v1/answer", json={"query": "A hotel question?"})
        self.assertEqual(response.status_code, 502)
        self.assertIn("error", response.json)
        self.assertNotIn("secret-provider-request-detail", response.get_data(as_text=True))

    def test_retrieval_endpoint_uses_four_contexts_by_default_without_generation(self):
        with patch.object(backend_app, "retrieve", return_value=["Policy evidence"]) as retrieve, patch.object(backend_app, "rag") as generate:
            response = self.client.post("/v1/relevant_chunks", json={"query": "A hotel question?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["relevant_chunks"], ["Policy evidence"])
        retrieve.assert_called_once_with("A hotel question?", k=4, vectorstore=self.store)
        generate.assert_not_called()

    def test_backend_functions_match_the_evaluated_notebook(self):
        notebook = json.loads((ROOT / "Full_Code_Notebook.ipynb").read_text(encoding="utf-8"))
        module = ast.parse((ROOT / "backend_files/rag_utils.py").read_text(encoding="utf-8"))
        exported = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        for number, name in ((73, "retrieve"), (90, "generate"), (93, "rag")):
            source = "".join(notebook["cells"][number - 1]["source"])
            if name == "rag":
                source = source.replace('config["OPENAI_API_KEY"]', 'os.environ["OPENAI_API_KEY"]').replace(
                    'config["OPENAI_API_BASE"]', 'os.environ["OPENAI_API_BASE"]')
            function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
            self.assertEqual(ast.dump(function, include_attributes=False), ast.dump(exported[name], include_attributes=False))


if __name__ == "__main__":
    unittest.main()
