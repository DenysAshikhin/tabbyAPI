import os
import unittest

from common.config_models import TabbyConfigModel
from common.tabby_config import TabbyConfig


class Exl3EnvOverrideTests(unittest.TestCase):
    """
    SiftKit launches TabbyAPI purely through TABBY_* environment overrides, so the contract that
    matters is: env string -> typed config field. Guards the upstream names adopted in the
    2026-08-26 merge (dynamic_draft, sysmem_kv_cache); a future upstream rename must fail here
    instead of silently ignoring the env vars.
    """

    ENV_KEYS = ("TABBY_DRAFT_MODEL_DYNAMIC_DRAFT", "TABBY_MEMORY_SYSMEM_KV_CACHE")

    def setUp(self):
        self.env_backup = {key: os.environ.get(key) for key in self.ENV_KEYS}

    def tearDown(self):
        for key, value in self.env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_environment_overrides_coerce_to_typed_config_fields(self):
        os.environ["TABBY_DRAFT_MODEL_DYNAMIC_DRAFT"] = "true"
        os.environ["TABBY_MEMORY_SYSMEM_KV_CACHE"] = "4096"

        config = TabbyConfigModel.model_validate(TabbyConfig()._from_environment())

        self.assertIs(config.draft_model.dynamic_draft, True)
        self.assertEqual(config.memory.sysmem_kv_cache, 4096)

    def test_defaults_leave_both_features_off(self):
        for key in self.ENV_KEYS:
            os.environ.pop(key, None)

        config = TabbyConfigModel.model_validate(TabbyConfig()._from_environment())

        self.assertIs(config.draft_model.dynamic_draft, False)
        self.assertEqual(config.memory.sysmem_kv_cache, 0)


if __name__ == "__main__":
    unittest.main()
