import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from common import auth, model
from endpoints.core.router import router as core_router


class ModelFreezeEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_auth_keys = auth.AUTH_KEYS
        self.original_disable_auth = auth.DISABLE_AUTH
        self.original_container = model.container

        auth.AUTH_KEYS = auth.AuthKeys(api_key="api", admin_key="admin")
        auth.DISABLE_AUTH = False
        self.app = FastAPI()
        self.app.include_router(core_router)

    async def asyncTearDown(self):
        auth.AUTH_KEYS = self.original_auth_keys
        auth.DISABLE_AUTH = self.original_disable_auth
        model.container = self.original_container

    async def _request(self, path, headers=None):
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(path, headers=headers)

    async def test_freeze_requires_admin_authentication(self):
        model.container = object()

        response = await self._request(
            "/v1/model/freeze", {"authorization": "Bearer api"}
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid admin key")

    async def test_restore_requires_admin_authentication(self):
        model.container = object()

        response = await self._request(
            "/v1/model/restore", {"authorization": "Bearer api"}
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid admin key")

    async def test_freeze_rejects_non_admin_before_checking_container(self):
        model.container = None

        response = await self._request(
            "/v1/model/freeze", {"authorization": "Bearer api"}
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid admin key")

    async def test_restore_rejects_non_admin_before_checking_container(self):
        model.container = None

        response = await self._request(
            "/v1/model/restore", {"authorization": "Bearer api"}
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid admin key")

    async def test_freeze_requires_a_loaded_model(self):
        model.container = None

        response = await self._request("/v1/model/freeze", {"x-admin-key": "admin"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("No models are currently loaded", response.json()["detail"])

    async def test_restore_requires_a_loaded_model(self):
        model.container = None

        response = await self._request("/v1/model/restore", {"x-admin-key": "admin"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("No models are currently loaded", response.json()["detail"])

    async def test_freeze_delegates_once_with_wait_skipped(self):
        model.container = object()

        with patch.object(model, "freeze_model_to_ram", new=AsyncMock()) as freeze:
            response = await self._request("/v1/model/freeze", {"x-admin-key": "admin"})

        self.assertEqual(response.status_code, 200)
        freeze.assert_awaited_once_with(skip_wait=True)

    async def test_restore_delegates_once(self):
        model.container = object()

        with patch.object(model, "restore_frozen_model", new=AsyncMock()) as restore:
            response = await self._request("/v1/model/restore", {"x-admin-key": "admin"})

        self.assertEqual(response.status_code, 200)
        restore.assert_awaited_once_with()

    async def test_offload_route_is_removed(self):
        model.container = object()

        response = await self._request("/v1/model/offload", {"x-admin-key": "admin"})

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
