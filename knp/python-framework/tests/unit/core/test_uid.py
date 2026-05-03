"""
@file test_uid.py
@brief UID tests.

@kaspersky_support OpenAI Codex.
@license Apache 2.0 License.
@copyright © 2026 AO Kaspersky Lab
@date 08.04.2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import uuid

import pytest

from knp.core import UID


class BrokenUUID(uuid.UUID):
    @property
    def bytes(self) -> bytes:
        return b"\x00" * 15


def test_uid_rejects_uuid_with_invalid_bytes_size() -> None:
    with pytest.raises(ValueError, match="exactly 16 bytes"):
        UID(BrokenUUID(bytes=b"\x00" * 16))
