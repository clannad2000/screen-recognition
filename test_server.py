import os
import io
import json
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from server import (
    ServiceError,
    add_optional_generation_params,
    extract_vlm_window_analysis,
    extract_vlm_summary,
    get_model_api_key,
    get_model_config,
    get_public_model_profiles,
    is_cloudflare_browser_signature_block,
    log_vlm_video_invalid_result,
    parse_vlm_video_job_config,
)


class CloudflareErrorTests(unittest.TestCase):
    def test_detects_cloudflare_browser_signature_ban(self):
        self.assertTrue(
            is_cloudflare_browser_signature_block(
                '{"error_code":1010,"error_name":"browser_signature_banned"}'
            )
        )

    def test_does_not_misclassify_other_errors(self):
        self.assertFalse(is_cloudflare_browser_signature_block('{"status":403}'))


class ApiKeyEnvironmentTests(unittest.TestCase):
    def test_loads_api_key_from_derived_environment_variable(self):
        with patch.dict(os.environ, {"HUGGING_FACE_API_KEY": "test-key"}, clear=True):
            self.assertEqual(
                get_model_api_key({"api_key_name": "hugging-face"}),
                ("HUGGING_FACE_API_KEY", "test-key"),
            )


class ModelProfileTests(unittest.TestCase):
    CONFIG = {
        "model_usage": "default",
        "model_profiles": {
            "default": {"model": "model-a", "chat_endpoint": "https://example.test/a"},
            "remote": {
                "model": "model-b",
                "chat_endpoint": "https://example.test/b",
                "provider": "test-provider",
                "api_key_name": "test-provider",
            },
        },
    }

    def test_uses_requested_profile_without_changing_default(self):
        self.assertEqual(get_model_config(self.CONFIG, "remote")["model"], "model-b")
        self.assertEqual(get_model_config(self.CONFIG)["model"], "model-a")

    def test_public_profiles_omit_sensitive_connection_details(self):
        with patch.dict(os.environ, {}, clear=True):
            profiles = get_public_model_profiles(self.CONFIG)
        self.assertEqual([profile["id"] for profile in profiles], ["default", "remote"])
        self.assertFalse(profiles[1]["apiKeyConfigured"])
        self.assertNotIn("chat_endpoint", profiles[1])


class GenerationParameterTests(unittest.TestCase):
    def test_omits_blank_generation_parameters(self):
        payload = {"model": "test-model"}
        add_optional_generation_params(
            payload, {"temperature": "", "max_tokens": ""}
        )
        self.assertNotIn("temperature", payload)
        self.assertNotIn("max_tokens", payload)

    def test_includes_configured_generation_parameters(self):
        payload = {"model": "test-model"}
        add_optional_generation_params(
            payload, {"temperature": 0.2, "max_tokens": 2048}
        )
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["max_tokens"], 2048)


class VlmVideoJobConfigTests(unittest.TestCase):
    def test_parses_browser_adjustable_parameters(self):
        parsed = parse_vlm_video_job_config(
            {
                "modelProfile": ["local-vlm"],
                "sampleInterval": ["5"],
                "windowSize": ["3"],
                "windowStep": ["1"],
                "startSec": ["10.5"],
                "endSec": ["90"],
            }
        )
        self.assertEqual(parsed["model_profile"], "local-vlm")
        self.assertEqual(parsed["window_size"], 3)
        self.assertEqual(parsed["start_sec"], 10.5)
        self.assertEqual(parsed["end_sec"], 90.0)

    def test_rejects_end_before_start(self):
        with self.assertRaises(ServiceError):
            parse_vlm_video_job_config(
                {
                    "modelProfile": ["local-vlm"],
                    "startSec": ["12"],
                    "endSec": ["10"],
                }
            )


class VlmSummaryTests(unittest.TestCase):
    def test_extracts_window_display_fields(self):
        result = extract_vlm_window_analysis(
            '{"summary":"窗口摘要","suspicious":true,"risk_level":"high","confidence":0.95}'
        )
        self.assertEqual(
            result,
            {
                "summary": "窗口摘要",
                "suspicious": True,
                "riskLevel": "high",
                "confidence": 0.95,
            },
        )

    def test_extracts_only_summary_from_fenced_json(self):
        content = '```json\n{"summary":"窗口摘要","events":[{"hidden":true}]}\n```'
        self.assertEqual(extract_vlm_summary(content), "窗口摘要")

    def test_rejects_missing_summary(self):
        with self.assertRaises(ServiceError):
            extract_vlm_summary('{"events":[]}')

    def test_logs_raw_content_when_summary_cannot_be_extracted(self):
        stream = io.StringIO()
        with redirect_stderr(stream):
            log_vlm_video_invalid_result(
                window_number=2,
                model_profile="test-vlm",
                frame_ids=["F000001", "F000002"],
                raw_content="模型返回的非 JSON 内容",
                error=ServiceError("模型未返回有效 JSON。"),
            )
        payload = json.loads(stream.getvalue())
        self.assertEqual(payload["event"], "vlm_video_invalid_model_result")
        self.assertEqual(payload["raw_model_content"], "模型返回的非 JSON 内容")


if __name__ == "__main__":
    unittest.main()
