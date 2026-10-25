"""Exercise the Streamlit question flow without contacting a backend/provider."""

import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
from streamlit.testing.v1 import AppTest
import requests

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "frontend_files/app.py"
CHUNK = "Content:\nSection: Pets\nPet fee: $50 per night, capped at $150 per stay.\n\nMetadata:\n{'source': 'Data/policy.pdf', 'sources': ['Data/policy.pdf'], 'section': 'Pets'}"


def response(payload):
    result = Mock(status_code=200)
    result.json.return_value = payload
    return result


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"BACKEND_URL": "http://policy.test:7860/"})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()

    def app(self):
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertFalse(app.exception)
        return app

    def test_initial_page_uses_evaluated_defaults_without_network(self):
        with patch("requests.post") as post:
            app = self.app()
        post.assert_not_called()
        self.assertEqual([item.value for item in app.slider], [4, 0.0, 512])
        self.assertEqual(app.radio[0].value, "Answer only")
        self.assertIn("ASH-CHI", app.text_area[0].placeholder)

    def test_empty_question_does_not_call_backend(self):
        with patch("requests.post") as post:
            app = self.app()
            app.button[0].click().run()
        post.assert_not_called()
        self.assertFalse(app.exception)
        self.assertIn("Please enter", app.warning[0].value)

    def test_answer_only_forwards_defaults_and_uses_bounded_timeout(self):
        with patch("requests.post", return_value=response({"answer": "The fee is $50 per night."})) as post:
            app = self.app()
            app.text_area[0].input("  What is the pet fee?  ")
            app.button[0].click().run()
        self.assertFalse(app.exception)
        post.assert_called_once_with("http://policy.test:7860/v1/answer", json={
            "query": "What is the pet fee?", "k": 4, "model_name": "gpt-4o-mini",
            "temperature": 0.0, "top_p": 1.0, "max_tokens": 512,
        }, timeout=(5, 90))
        self.assertIn("$50", " ".join(item.value for item in app.markdown))

    def test_sources_show_policy_text_and_filename_without_internal_metadata(self):
        with patch("requests.post", return_value=response({"answer": "The fee is $50 per night.", "relevant_chunks": [CHUNK]})) as post:
            app = self.app()
            app.text_area[0].input("What is the pet fee?")
            app.radio[0].set_value("Answer with sources")
            app.slider[0].set_value(2)
            app.slider[1].set_value(0.3)
            app.slider[2].set_value(1024)
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(post.call_args.args[0].endswith("/v1/answer_with_relevant_chunks"))
        self.assertEqual(post.call_args.kwargs["json"]["k"], 2)
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], 0.3)
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 1024)
        self.assertTrue(any("policy.pdf" in item.value for item in app.caption))
        self.assertNotIn("Metadata:", " ".join(item.value for item in app.markdown))

    def test_changing_settings_keeps_answer_without_resubmitting(self):
        with patch("requests.post", return_value=response({"answer": "Saved policy answer."})) as post:
            app = self.app()
            app.text_area[0].input("Hotel question?")
            app.button[0].click().run()
            app.slider[0].set_value(3).run()
        self.assertFalse(app.exception)
        self.assertEqual(post.call_count, 1)
        self.assertIn("Saved policy answer.", " ".join(item.value for item in app.markdown))

    def test_timeout_clears_previous_answer_and_shows_helpful_error(self):
        with patch("requests.post", side_effect=[response({"answer": "Old answer."}), requests.Timeout("private connection detail")]):
            app = self.app()
            app.text_area[0].input("First question?")
            app.button[0].click().run()
            app.text_area[0].input("Second question?")
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("longer than expected", app.error[0].value)
        self.assertNotIn("private", app.error[0].value)
        self.assertIsNone(app.session_state["policy_result"])

    def test_backend_http_errors_are_handled_without_raw_details(self):
        bad = response({"error": "private provider detail"})
        bad.raise_for_status.side_effect = requests.HTTPError("private provider detail")
        with patch("requests.post", return_value=bad):
            app = self.app()
            app.text_area[0].input("Hotel question?")
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("could not answer", app.error[0].value)
        self.assertNotIn("private", app.error[0].value)

    def test_malformed_response_is_reported_without_app_exception(self):
        with patch("requests.post", return_value=response({"answer": 123})):
            app = self.app()
            app.text_area[0].input("Hotel question?")
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("unexpected response", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
