"""
@file test_executor.py
@kaspersky_support Artiom N.
@license Apache 2.0
@copyright © 2025 AO Kaspersky Lab

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



from knp.base_framework import Network, Model, ModelExecutor, BackendLoader
from knp.core import BLIFATNeuronPopulation, DeltaSynapseProjection, UID
from knp.neuron_traits import BLIFATNeuronParameters
from knp.synapse_traits import DeltaSynapseParameters, OutputType


def neuron_generator(_):  # type: ignore[no-untyped-def]
    return BLIFATNeuronParameters()


def synapse_generator(_):  # type: ignore[no-untyped-def]
    return DeltaSynapseParameters(1.0, 6, OutputType.EXCITATORY), 0, 0


def input_projection_gen(_):  # type: ignore[no-untyped-def]
    return DeltaSynapseParameters(1.0, 1, OutputType.EXCITATORY), 0, 0


def create_model():  # type: ignore[no-untyped-def]
    main_population = BLIFATNeuronPopulation(neuron_generator, 1)
    loop_projection = DeltaSynapseProjection(main_population.uid, main_population.uid, synapse_generator, 1)
    input_projection = DeltaSynapseProjection(UID(False), main_population.uid, input_projection_gen, 1)

    network = Network()
    network.add_population(main_population)
    network.add_projection(input_projection)
    network.add_projection(loop_projection)

    model = Model(network)
    input_channel_uid, output_channel_uid = UID(), UID()
    model.add_input_channel(input_channel_uid, input_projection.uid)
    model.add_output_channel(output_channel_uid, main_population.uid)
    return model, input_channel_uid, output_channel_uid


def input_data_generator(step_num):  # type: ignore[no-untyped-def]
    if step_num % 5 == 0:
        return [0]
    else:
        return []


def is_continue_execution(step_num):  # type: ignore[no-untyped-def]
    return step_num < 20


def run_model(model, input_channel_uid, input_generator, output_uid, root_path):  # type: ignore[no-untyped-def]
    with BackendLoader() as loader:
        backend = loader.load(f'{root_path}/../bin/knp-cpu-single-threaded-backend')
        input_channel_map = {input_channel_uid: input_generator}
        model_executor = ModelExecutor(model, backend, input_channel_map)
        model_executor.start(is_continue_execution)
        output = model_executor.get_output_channel(output_uid).read_some_from_buffer(0, 20)
        return output


def messages_to_results(messages):  # type: ignore[no-untyped-def]
    result = []
    for msg in messages:
        result.append(msg.header.send_time)
    return result


def test_small_network(pytestconfig):  # type: ignore[no-untyped-def]
    model, input_uid, output_uid = create_model()  # type: ignore[no-untyped-call]
    messages = run_model(model, input_uid, input_data_generator, output_uid, pytestconfig.rootdir)  # type: ignore[no-untyped-call]
    results = messages_to_results(messages)  # type: ignore[no-untyped-call]
    expected_results = [1, 6, 7, 11, 12, 13, 16, 17, 18, 19]
    assert results == expected_results
