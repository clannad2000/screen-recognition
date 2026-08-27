import os
import unittest
from unittest.mock import patch

from server import (
    get_model_api_key,
    get_model_config,
    get_public_model_profiles,
    is_cloudflare_browser_signature_block,
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


if __name__ == "__main__":
    unittest.main()
