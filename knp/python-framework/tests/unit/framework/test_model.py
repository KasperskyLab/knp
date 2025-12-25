"""
@file test_model.py
@brief Model tests.

@kaspersky_support Vartenkov A.
@license Apache 2.0 License.
@copyright © 2024 AO Kaspersky Lab
@date 24.10.2024.

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

from knp.base_framework import Network, Model
from knp.core import UID


def test_add_empty_network() -> None:
    network_uid = UID()
    network = Network(network_uid)
    model = Model(network)
    assert model.network.populations_count == 0
    assert model.network.projections_count == 0
    assert model.network.get_uid() == network_uid
