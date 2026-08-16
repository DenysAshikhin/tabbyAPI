import asyncio
import unittest

from backends.exllamav3.model import ExllamaV3Container
from common import model as model_module


class _PublicModelDouble:
    def __init__(
        self,
        name,
        events,
        *,
        fail_freeze=False,
        fail_load=False,
        cancel_load=False,
        fail_unload=False,
        fail_unload_once=False,
    ):
        self.name = name
        self.events = events
        self.source = object()
        self.fail_freeze = fail_freeze
        self.fail_load = fail_load
        self.cancel_load = cancel_load
        self.fail_unload = fail_unload
        self.fail_unload_once = fail_unload_once
        self.load_kwargs = []

    @property
    def modules(self):
        raise AssertionError("Tabby must not enumerate EXL3 modules")

    @property
    def config(self):
        raise AssertionError("Tabby must not access EXL3 config internals")

    def get_tensors(self):
        raise AssertionError("Tabby must not inspect EXL3 tensors")

    def freeze(self):
        self.events.append(f"freeze:{self.name}")
        if self.fail_freeze:
            raise RuntimeError(f"freeze failed: {self.name}")
        return self.source

    def unload(self):
        self.events.append(f"unload:{self.name}")
        if self.fail_unload_once:
            self.fail_unload_once = False
            raise RuntimeError(f"unload failed once: {self.name}")
        if self.fail_unload:
            raise RuntimeError(f"unload failed: {self.name}")

    def load_gen(self, **kwargs):
        self.load_kwargs.append(kwargs)
        self.events.append((f"load:{self.name}", kwargs["source"]))
        if self.cancel_load:
            raise asyncio.CancelledError()
        if self.fail_load:
            raise RuntimeError(f"load failed: {self.name}")
        return iter(())


class _CacheDouble:
    def __init__(self, name, events, *, fail_detach=False):
        self.name = name
        self.events = events
        self.fail_detach = fail_detach

    def detach_from_model(self):
        self.events.append(f"detach:{self.name}")
        if self.fail_detach:
            raise RuntimeError(f"detach failed: {self.name}")


class _GeneratorDouble:
    def __init__(self, events, *, fail_close=False, close_started=None, close_release=None):
        self.events = events
        self.fail_close = fail_close
        self.close_started = close_started
        self.close_release = close_release

    async def close(self):
        self.events.append("generator-close")
        if self.close_started is not None:
            self.close_started.set()
        if self.close_release is not None:
            await self.close_release.wait()
        if self.fail_close:
            raise RuntimeError("generator close failed")


class _FreezeContainer(ExllamaV3Container):
    def __init__(self):
        super().__init__()
        self.restore_events = []
        self.fail_generator = False
        self.close_started = None
        self.close_release = None

    def create_cache(self, mode, model):
        self.restore_events.append(("cache", mode, model.name))
        return _CacheDouble(model.name, self.restore_events)

    async def create_generator(self):
        self.restore_events.append("generator")
        if self.fail_generator:
            self.generator = _GeneratorDouble(
                self.restore_events,
                close_started=self.close_started,
                close_release=self.close_release,
            )
            raise RuntimeError("generator failed")


class Exl3ComponentInventoryTests(unittest.TestCase):
    def test_component_inventory_includes_each_present_component_once_in_order(self):
        container = ExllamaV3Container()
        main = _PublicModelDouble("main", [])
        draft = _PublicModelDouble("draft", [])
        vision = _PublicModelDouble("vision", [])
        container.model = main
        container.draft_model = draft
        container.vision_model = vision

        inventory = container._component_inventory()

        self.assertEqual(
            inventory,
            (("vision", vision), ("draft", draft), ("main", main)),
        )

    def test_component_inventory_skips_missing_optional_components(self):
        container = ExllamaV3Container()
        main = _PublicModelDouble("main", [])
        container.model = main

        self.assertEqual(container._component_inventory(), (("main", main),))


class Exl3FreezeResidencyTests(unittest.IsolatedAsyncioTestCase):
    def _container_with_components(self):
        events = []
        container = _FreezeContainer()
        container.model = _PublicModelDouble("main", events)
        container.draft_model = _PublicModelDouble("draft", events)
        container.vision_model = _PublicModelDouble("vision", events)
        container.use_draft_model = True
        container.use_vision = True
        return container, events

    async def test_freeze_freezes_and_unloads_components_without_internal_tensor_access(self):
        container, events = self._container_with_components()

        await container.freeze_to_ram()

        self.assertEqual(
            events,
            [
                "freeze:vision",
                "freeze:draft",
                "freeze:main",
                "unload:vision",
                "unload:draft",
                "unload:main",
            ],
        )
        self.assertEqual(
            container.frozen_sources,
            {
                "vision": container.vision_model.source,
                "draft": container.draft_model.source,
                "main": container.model.source,
            },
        )
        self.assertFalse(hasattr(container, "ram_tensors"))

    async def test_restore_loads_each_component_from_its_public_source(self):
        container, events = self._container_with_components()
        container.frozen_sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }

        await container.restore_from_freeze()

        self.assertEqual(
            events,
            [
                ("load:vision", container.vision_model.source),
                ("load:draft", container.draft_model.source),
                ("load:main", container.model.source),
            ],
        )
        self.assertEqual(
            container.restore_events,
            [
                ("cache", container.cache_mode, "main"),
                ("cache", container.draft_cache_mode, "draft"),
                "generator",
            ],
        )
        self.assertIsNone(container.frozen_sources)

    async def test_restore_uses_explicit_single_device_source_loads(self):
        container, _ = self._container_with_components()
        container.frozen_sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }

        await container.restore_from_freeze()

        for model in (container.vision_model, container.draft_model, container.model):
            self.assertEqual(
                model.load_kwargs,
                [{"callback": None, "source": model.source, "device": "cuda:0"}],
            )

    async def test_freeze_failure_keeps_runtime_intact_without_partial_sources(self):
        container, events = self._container_with_components()
        container.draft_model.fail_freeze = True
        original_generator = object()
        original_cache = object()
        original_draft_cache = object()
        container.generator = original_generator
        container.cache = original_cache
        container.draft_cache = original_draft_cache
        container.loaded = True

        with self.assertRaisesRegex(RuntimeError, "freeze failed: draft"):
            await container.freeze_to_ram()

        self.assertEqual(events, ["freeze:vision", "freeze:draft"])
        self.assertIs(container.generator, original_generator)
        self.assertIs(container.cache, original_cache)
        self.assertIs(container.draft_cache, original_draft_cache)
        self.assertTrue(container.loaded)
        self.assertIsNone(container.frozen_sources)

    async def test_freeze_missing_source_keeps_runtime_intact_without_partial_sources(self):
        container, events = self._container_with_components()
        container.model.source = None
        original_generator = object()
        original_cache = object()
        container.generator = original_generator
        container.cache = original_cache
        container.loaded = True

        with self.assertRaisesRegex(RuntimeError, "source for main"):
            await container.freeze_to_ram()

        self.assertEqual(
            events,
            ["freeze:vision", "freeze:draft", "freeze:main"],
        )
        self.assertIs(container.generator, original_generator)
        self.assertIs(container.cache, original_cache)
        self.assertTrue(container.loaded)
        self.assertIsNone(container.frozen_sources)

    async def test_freeze_teardown_failure_can_retry_without_recapturing_sources(self):
        container, events = self._container_with_components()
        container.vision_model.fail_unload_once = True
        container.loaded = True

        with self.assertRaisesRegex(RuntimeError, "unload failed once: vision"):
            await container.freeze_to_ram()

        sources = container.frozen_sources
        self.assertIsNotNone(sources)
        self.assertFalse(container.loaded)
        self.assertEqual(
            events,
            [
                "freeze:vision",
                "freeze:draft",
                "freeze:main",
                "unload:vision",
                "unload:draft",
                "unload:main",
            ],
        )

        await container.freeze_to_ram()

        self.assertIs(container.frozen_sources, sources)
        self.assertEqual(
            events,
            [
                "freeze:vision",
                "freeze:draft",
                "freeze:main",
                "unload:vision",
                "unload:draft",
                "unload:main",
                "unload:vision",
                "unload:draft",
                "unload:main",
            ],
        )

    async def test_restore_load_failure_unloads_partial_state_and_retains_sources(self):
        container, events = self._container_with_components()
        container.draft_model.fail_load = True
        sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }
        container.frozen_sources = sources

        with self.assertRaisesRegex(RuntimeError, "load failed: draft"):
            await container.restore_from_freeze()

        self.assertEqual(
            events,
            [
                ("load:vision", sources["vision"]),
                ("load:draft", sources["draft"]),
                "unload:vision",
                "unload:draft",
                "unload:main",
            ],
        )
        self.assertEqual(
            container.restore_events,
            [
                ("cache", container.cache_mode, "main"),
                ("cache", container.draft_cache_mode, "draft"),
                "detach:main",
                "detach:draft",
            ],
        )
        self.assertIsNone(container.generator)
        self.assertFalse(container.loaded)
        self.assertIs(container.frozen_sources, sources)
        self.assertIsNone(container.cache)
        self.assertIsNone(container.draft_cache)

    async def test_restore_late_main_load_failure_unloads_every_component_and_retains_sources(self):
        container, events = self._container_with_components()
        container.model.fail_load = True
        sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }
        container.frozen_sources = sources

        with self.assertRaisesRegex(RuntimeError, "load failed: main"):
            await container.restore_from_freeze()

        self.assertEqual(
            events,
            [
                ("load:vision", sources["vision"]),
                ("load:draft", sources["draft"]),
                ("load:main", sources["main"]),
                "unload:vision",
                "unload:draft",
                "unload:main",
            ],
        )
        self.assertIs(container.frozen_sources, sources)
        self.assertFalse(container.loaded)
        self.assertIsNone(container.cache)
        self.assertIsNone(container.draft_cache)

    async def test_restore_cancellation_unloads_every_component_and_retains_sources(self):
        container, events = self._container_with_components()
        container.draft_model.cancel_load = True
        sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }
        container.frozen_sources = sources

        with self.assertRaises(asyncio.CancelledError):
            await container.restore_from_freeze()

        self.assertEqual(
            events,
            [
                ("load:vision", sources["vision"]),
                ("load:draft", sources["draft"]),
                "unload:vision",
                "unload:draft",
                "unload:main",
            ],
        )
        self.assertIs(container.frozen_sources, sources)
        self.assertFalse(container.loaded)

    async def test_restore_requires_exact_current_inventory_sources_without_loading(self):
        container, events = self._container_with_components()
        sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "extra": object(),
        }
        container.frozen_sources = sources

        with self.assertRaisesRegex(RuntimeError, "exact"):
            await container.restore_from_freeze()

        self.assertEqual(
            events,
            ["unload:vision", "unload:draft", "unload:main"],
        )
        self.assertEqual(container.restore_events, [])
        self.assertIs(container.frozen_sources, sources)
        self.assertFalse(container.loaded)

    async def test_restore_without_frozen_state_does_not_mutate_loaded_runtime(self):
        container, events = self._container_with_components()
        generator = _GeneratorDouble(events)
        cache = object()
        draft_cache = object()
        container.generator = generator
        container.cache = cache
        container.draft_cache = draft_cache
        container.loaded = True

        with self.assertRaisesRegex(RuntimeError, "No frozen weights"):
            await container.restore_from_freeze()

        self.assertIs(container.generator, generator)
        self.assertIs(container.cache, cache)
        self.assertIs(container.draft_cache, draft_cache)
        self.assertTrue(container.loaded)
        self.assertIsNone(container.frozen_sources)
        self.assertEqual(events, [])

    async def test_restore_generator_failure_cleans_runtime_and_retains_sources(self):
        container, events = self._container_with_components()
        sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }
        container.frozen_sources = sources
        container.fail_generator = True

        with self.assertRaisesRegex(RuntimeError, "generator failed"):
            await container.restore_from_freeze()

        self.assertEqual(
            events,
            [
                ("load:vision", sources["vision"]),
                ("load:draft", sources["draft"]),
                ("load:main", sources["main"]),
                "unload:vision",
                "unload:draft",
                "unload:main",
            ],
        )
        self.assertEqual(
            container.restore_events,
            [
                ("cache", container.cache_mode, "main"),
                ("cache", container.draft_cache_mode, "draft"),
                "generator",
                "generator-close",
                "detach:main",
                "detach:draft",
            ],
        )
        self.assertIsNone(container.generator)
        self.assertFalse(container.loaded)
        self.assertIs(container.frozen_sources, sources)
        self.assertIsNone(container.cache)
        self.assertIsNone(container.draft_cache)

    async def test_restore_rollback_keeps_lock_until_cleanup_finishes(self):
        container, events = self._container_with_components()
        sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }
        container.frozen_sources = sources
        container.fail_generator = True
        container.close_started = asyncio.Event()
        container.close_release = asyncio.Event()

        restore = asyncio.create_task(container.restore_from_freeze())
        await container.close_started.wait()
        freeze = asyncio.create_task(container.freeze_to_ram())
        unload = asyncio.create_task(container.unload())
        await asyncio.sleep(0)

        self.assertFalse(
            any(isinstance(event, str) and event.startswith("freeze:") for event in events)
        )
        self.assertFalse(
            any(isinstance(event, str) and event.startswith("unload:") for event in events)
        )
        self.assertTrue(container.load_lock.locked())

        container.close_release.set()
        with self.assertRaisesRegex(RuntimeError, "generator failed"):
            await restore
        await asyncio.gather(freeze, unload)
        self.assertFalse(container.load_lock.locked())

    async def test_full_unload_clears_all_state_after_cleanup_errors(self):
        container, events = self._container_with_components()
        container.generator = _GeneratorDouble(events, fail_close=True)
        container.cache = _CacheDouble("main", events, fail_detach=True)
        container.draft_cache = _CacheDouble("draft", events)
        container.draft_model.fail_unload = True
        container.frozen_sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }
        container.loaded = True

        with self.assertRaisesRegex(RuntimeError, "generator close failed"):
            await container.unload()

        self.assertEqual(
            events,
            [
                "generator-close",
                "unload:main",
                "unload:draft",
                "unload:vision",
                "detach:main",
                "detach:draft",
            ],
        )
        self.assertIsNone(container.generator)
        self.assertIsNone(container.cache)
        self.assertIsNone(container.draft_cache)
        self.assertIsNone(container.model)
        self.assertIsNone(container.draft_model)
        self.assertIsNone(container.vision_model)
        self.assertIsNone(container.config)
        self.assertIsNone(container.draft_config)
        self.assertIsNone(container.tokenizer)
        self.assertIsNone(container.frozen_sources)
        self.assertFalse(container.loaded)

    async def test_freeze_cancellation_waiting_for_lock_preserves_other_owner(self):
        container, _ = self._container_with_components()
        await container.load_lock.acquire()

        operation = asyncio.create_task(container.freeze_to_ram())
        await asyncio.sleep(0)
        operation.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await operation

        self.assertTrue(container.load_lock.locked())
        container.load_lock.release()

    async def test_restore_cancellation_waiting_for_lock_preserves_other_owner(self):
        container, _ = self._container_with_components()
        container.frozen_sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }
        original_cache = object()
        original_draft_cache = object()
        container.cache = original_cache
        container.draft_cache = original_draft_cache
        container.loaded = True
        await container.load_lock.acquire()

        operation = asyncio.create_task(container.restore_from_freeze())
        await asyncio.sleep(0)
        operation.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await operation

        self.assertTrue(container.load_lock.locked())
        self.assertIs(container.cache, original_cache)
        self.assertIs(container.draft_cache, original_draft_cache)
        self.assertTrue(container.loaded)
        container.load_lock.release()

    async def test_freeze_teardown_continues_after_failure_and_retains_sources(self):
        container, events = self._container_with_components()
        container.draft_model.fail_unload = True
        container.generator = _GeneratorDouble(events)
        container.cache = _CacheDouble("main", events, fail_detach=True)
        container.draft_cache = _CacheDouble("draft", events)
        container.loaded = True

        with self.assertRaisesRegex(RuntimeError, "unload failed: draft"):
            await container.freeze_to_ram()

        self.assertEqual(
            events,
            [
                "freeze:vision",
                "freeze:draft",
                "freeze:main",
                "generator-close",
                "unload:vision",
                "unload:draft",
                "unload:main",
                "detach:main",
                "detach:draft",
            ],
        )
        self.assertIsNone(container.generator)
        self.assertIsNone(container.cache)
        self.assertIsNone(container.draft_cache)
        self.assertFalse(container.loaded)
        self.assertEqual(
            container.frozen_sources,
            {
                "vision": container.vision_model.source,
                "draft": container.draft_model.source,
                "main": container.model.source,
            },
        )

    async def test_freeze_generator_close_failure_still_cleans_and_retains_sources(self):
        container, events = self._container_with_components()
        container.generator = _GeneratorDouble(events, fail_close=True)
        container.cache = _CacheDouble("main", events)
        container.draft_cache = _CacheDouble("draft", events)
        container.loaded = True

        with self.assertRaisesRegex(RuntimeError, "generator close failed"):
            await container.freeze_to_ram()

        self.assertEqual(
            events,
            [
                "freeze:vision",
                "freeze:draft",
                "freeze:main",
                "generator-close",
                "unload:vision",
                "unload:draft",
                "unload:main",
                "detach:main",
                "detach:draft",
            ],
        )
        self.assertIsNone(container.generator)
        self.assertIsNone(container.cache)
        self.assertIsNone(container.draft_cache)
        self.assertFalse(container.loaded)
        self.assertIsNotNone(container.frozen_sources)

    async def test_full_unload_releases_retained_frozen_sources(self):
        container, _ = self._container_with_components()
        container.frozen_sources = {
            "vision": container.vision_model.source,
            "draft": container.draft_model.source,
            "main": container.model.source,
        }

        await container.unload()

        self.assertIsNone(container.frozen_sources)

    async def test_full_unload_releases_components_main_to_vision(self):
        container, events = self._container_with_components()

        await container.unload()

        self.assertEqual(
            events,
            ["unload:main", "unload:draft", "unload:vision"],
        )


class _CommonContainer:
    def __init__(self):
        self.calls = []

    async def freeze_to_ram(self, **kwargs):
        self.calls.append(("freeze", kwargs))

    async def restore_from_freeze(self):
        self.calls.append(("restore", {}))


class ModelFreezeResidencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_container = model_module.container

    async def asyncTearDown(self):
        model_module.container = self.original_container

    async def test_common_model_delegates_freeze_residency_operations(self):
        container = _CommonContainer()
        model_module.container = container

        await model_module.freeze_model_to_ram(skip_wait=True)
        await model_module.restore_frozen_model()

        self.assertEqual(
            container.calls,
            [("freeze", {"skip_wait": True}), ("restore", {})],
        )

    async def test_common_model_fails_loudly_without_a_container(self):
        model_module.container = None

        with self.assertRaisesRegex(RuntimeError, "No model is available to freeze"):
            await model_module.freeze_model_to_ram()
        with self.assertRaisesRegex(RuntimeError, "No frozen model is available to restore"):
            await model_module.restore_frozen_model()


if __name__ == "__main__":
    unittest.main()
