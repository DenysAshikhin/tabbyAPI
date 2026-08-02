import os
import unittest

from backends.exllamav3.model import ExllamaV3Container
from common.config_models import TabbyConfigModel
from common.tabby_config import TabbyConfig


class Exl3DraftDynamicAndPageCacheTests(unittest.TestCase):
    """
    SiftKit launches TabbyAPI purely through TABBY_* environment overrides, so the contract that
    matters is: env string -> typed config field -> container/generator knob.
    """

    def setUp(self):
        self.env_backup = {
            key: os.environ.get(key)
            for key in ("TABBY_DRAFT_MODEL_DRAFT_DYNAMIC", "TABBY_MEMORY_SYSMEM_PAGE_CACHE")
        }

    def tearDown(self):
        for key, value in self.env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_environment_overrides_coerce_to_typed_config_fields(self):
        os.environ["TABBY_DRAFT_MODEL_DRAFT_DYNAMIC"] = "true"
        os.environ["TABBY_MEMORY_SYSMEM_PAGE_CACHE"] = "4096"

        config = TabbyConfigModel.model_validate(TabbyConfig()._from_environment())

        self.assertIs(config.draft_model.draft_dynamic, True)
        self.assertEqual(config.memory.sysmem_page_cache, 4096)

    def test_defaults_leave_both_features_off(self):
        os.environ.pop("TABBY_DRAFT_MODEL_DRAFT_DYNAMIC", None)
        os.environ.pop("TABBY_MEMORY_SYSMEM_PAGE_CACHE", None)

        config = TabbyConfigModel.model_validate(TabbyConfig()._from_environment())

        self.assertIs(config.draft_model.draft_dynamic, False)
        self.assertEqual(config.memory.sysmem_page_cache, 0)

    def test_mtp_drafting_carries_the_dynamic_flag_and_token_ceiling(self):
        container = ExllamaV3Container()

        container.configure_drafting(
            {"draft_mode": "mtp", "draft_num_tokens": 5, "draft_dynamic": True}
        )

        self.assertTrue(container.use_draft_model)
        self.assertIs(container.draft_dynamic, True)
        self.assertEqual(container.draft_num_tokens, 5)

    def test_dynamic_drafting_is_off_without_a_draft_model(self):
        container = ExllamaV3Container()

        container.configure_drafting({"draft_mode": "disabled", "draft_dynamic": True})

        self.assertFalse(container.use_draft_model)
        self.assertIs(container.draft_dynamic, False)
        self.assertIsNone(container.draft_num_tokens)

    def test_ngram_drafting_does_not_enable_dynamic_windows(self):
        container = ExllamaV3Container()

        container.configure_drafting(
            {"draft_mode": "ngram", "ngram_match_min": 2, "draft_dynamic": True}
        )

        self.assertFalse(container.use_draft_model)
        self.assertIs(container.draft_dynamic, False)

    def test_unknown_draft_mode_is_rejected(self):
        container = ExllamaV3Container()

        with self.assertRaises(ValueError):
            container.configure_drafting({"draft_mode": "nonsense"})


if __name__ == "__main__":
    unittest.main()
