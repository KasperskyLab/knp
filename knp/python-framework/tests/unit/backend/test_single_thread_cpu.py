"""
@file test_single_thread_cpu.py
@brief Kaspersky Neuromorphic Platform single-thread CPU backend tests.

@kaspersky_support Artiom N.
@license Apache 2.0 License.
@copyright © 2024 AO Kaspersky Lab
@date 01.03.2024.

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


from knp.base_framework import BackendLoader
from knp.core import UID, BLIFATNeuronPopulation, DeltaSynapseProjection
from knp.core.messaging import SpikeMessage
from knp.neuron_traits import BLIFATNeuronParameters
from knp.synapse_traits import DeltaSynapseParameters, OutputType


def neuron_generator(_):  # type: ignore[no-untyped-def]
    return BLIFATNeuronParameters()


def synapse_generator(_):  # type: ignore[no-untyped-def]
    return DeltaSynapseParameters(1.0, 6, OutputType.EXCITATORY), 0, 0


def input_projection_gen(_):  # type: ignore[no-untyped-def]
    return DeltaSynapseParameters(1.0, 1, OutputType.EXCITATORY), 0, 0


def test_smallest_network(pytestconfig):  # type: ignore[no-untyped-def]
    population = BLIFATNeuronPopulation(neuron_generator, 1)

    loop_projection = DeltaSynapseProjection(population.uid, population.uid, synapse_generator, 1)
    input_projection = DeltaSynapseProjection(UID(False), population.uid, input_projection_gen, 1)

    input_uid = input_projection.uid

    backend = BackendLoader().load(f'{pytestconfig.rootdir}/../bin/knp-cpu-single-threaded-backend')

    backend.load_all_populations([population])
    backend.load_all_projections([input_projection, loop_projection])

    backend._init()
    endpoint = backend.message_bus.create_endpoint()

    in_channel_uid = UID()
    out_channel_uid = UID()

    backend.subscribe(SpikeMessage, input_uid, [in_channel_uid])
    endpoint.subscribe(SpikeMessage, out_channel_uid, [population.uid])

    print(f'POP UID = {population.uid}, IC UID = {in_channel_uid}')

    results = []

    for step in range(0, 20):
        # Send inputs on steps 0, 5, 10, 15.
        if step % 5 == 0:
            print(f'STEP {step}')
            message = SpikeMessage((in_channel_uid, step), [0])
            endpoint.send_message(message)

        backend._step()
        messages_count = endpoint.receive_all_messages()
        print(f'Received {messages_count} messages.')
        output = endpoint.unload_messages(SpikeMessage, out_channel_uid)
        # Write the steps on which the network sends a spike.
        if len(output):
            results.append(step)

    # Spikes on steps "5n + 1" (input) and on "previous_spike_n + 6" (positive feedback loop).
    expected_results = [1, 6, 7, 11, 12, 13, 16, 17, 18, 19]

    assert results == expected_results
