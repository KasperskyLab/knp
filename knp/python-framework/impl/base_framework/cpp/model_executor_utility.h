/**
 * @file model_executor_utility.h
 * @brief Auxiliary functions for ModelExecutor class bindings.
 * @kaspersky_support Vartenkov A.
 * @date 31.10.2024
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <knp/core/backend.h>
#include <knp/framework/model_executor.h>

#include <memory>
#include <utility>
#include <vector>


template <class DataOut, class DataIn>
struct BinaryFunction
{
public:
    explicit BinaryFunction(py::object func) : func_(std::move(func))
    {
        try
        {
            py::object call = func_.attr("__call__");
        }
        catch (...)
        {
            throw;
        }
    }
    //knp::core::messaging::SpikeData operator()(const knp::core::Step &step)
    DataOut operator()(const DataIn &input) { return boost::python::call<DataOut>(func_.ptr(), input); }

private:
    py::object func_;
};


template <class DataIn>
struct BinaryFunction<knp::core::messaging::SpikeData, DataIn>
{
    using DataOut = knp::core::messaging::SpikeData;

public:
    explicit BinaryFunction(py::object func) : func_(std::move(func))
    {
        try
        {
            py::object call = func_.attr("__call__");
        }
        catch (...)
        {
            throw;
        }
    }

    DataOut operator()(const DataIn &input)
    {
        auto value = py::call<py::list>(func_.ptr(), input);
        return DataOut(
            py::stl_input_iterator<knp::core::messaging::SpikeIndex>(value),
            py::stl_input_iterator<knp::core::messaging::SpikeIndex>());
    }

private:
    py::object func_;
};


std::shared_ptr<knp::framework::ModelExecutor> create_model_executor(
    knp::framework::Model &model, std::shared_ptr<knp::core::Backend> &backend, py::dict &input_map)
{
    knp::framework::ModelLoader::InputChannelMap i_map;
    py::list keys = input_map.keys();
    for (int64_t i = 0; i < py::len(keys); ++i)
    {
        knp::core::UID uid = py::extract<knp::core::UID>(keys[i]);
        py::object value = input_map.get(keys[i]);
        BinaryFunction<knp::core::messaging::SpikeData, knp::core::Step> channel_function{value};
        i_map.insert({uid, channel_function});
    }
    return std::make_shared<knp::framework::ModelExecutor>(model, backend, i_map);
}


void start_model_executor(knp::framework::ModelExecutor &self)
{
    self.start();
}


void start_model_executor_predicate(knp::framework::ModelExecutor &self, const py::object &function)
{
    BinaryFunction<bool, knp::core::Step> predicate{function};
    self.start(predicate);
}


void add_executor_spike_observer(
    knp::framework::ModelExecutor &self, const py::object &message_proc, const std::vector<knp::core::UID> &senders)
{
    self.add_observer(
        std::function{BinaryFunction<void, std::vector<knp::core::messaging::SpikeMessage>>{message_proc}}, senders);
}


void add_executor_impact_observer(
    knp::framework::ModelExecutor &self, const py::object &impact_proc, const std::vector<knp::core::UID> &senders)
{
    self.add_observer(
        std::function{BinaryFunction<void, std::vector<knp::core::messaging::SynapticImpactMessage>>{impact_proc}},
        senders);
}


auto &get_output_channel(knp::framework::ModelExecutor &self, const knp::core::UID &channel_uid)
{
    return self.get_loader().get_output_channel(channel_uid);
}
