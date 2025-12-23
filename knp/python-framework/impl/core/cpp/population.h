/**
 * @file population.h
 * @brief Python bindings header for Population.
 * @kaspersky_support Artiom N.
 * @date 07.02.2024
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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

#include <knp/core/population.h>
#include <knp/neuron-traits/all_traits.h>

#include <iostream>
#include <memory>
#include <optional>

#include "common.h"

template <typename ElemParametersType>
struct PopulationGeneratorProxy
{
    explicit PopulationGeneratorProxy(const py::object &gen_func) : gen_func_(gen_func)
    {
        if (!static_cast<bool>(PyCallable_Check(gen_func.ptr())))
        {
            PyErr_SetString(PyExc_TypeError, "Passed generator is not callable.");
            py::throw_error_already_set();
        }
    }

    std::optional<ElemParametersType> operator()(size_t index)
    {
        auto res = py::call<py::object>(gen_func_.ptr(), index);
        if (res.is_none())
        {
            return std::nullopt;
        }

        return py::extract<ElemParametersType>(res);
    }

    const py::object &gen_func_;
};


template <typename ElemType>
std::shared_ptr<core::Population<ElemType>> population_constructor_wrapper(
    const core::UID &uid, const py::object &gen_func, size_t index)
{
    return std::make_shared<core::Population<ElemType>>(
        uid, PopulationGeneratorProxy<typename core::Population<ElemType>::NeuronParameters>(gen_func), index);
};


template <typename ElemType>
std::shared_ptr<core::Population<ElemType>> population_constructor_wrapper(const py::object &gen_func, size_t index)
{
    return std::make_shared<core::Population<ElemType>>(
        PopulationGeneratorProxy<typename core::Population<ElemType>::NeuronParameters>(gen_func), index);
};


template <typename ElemType>
void population_neurons_add_wrapper(core::Population<ElemType> &population, const py::object &gen_func, size_t index)
{
    population.add_neurons(
        PopulationGeneratorProxy<typename core::Population<ElemType>::NeuronParameters>(gen_func), index);
};


namespace nt = knp::neuron_traits;

#define INSTANCE_POPULATION_CONVERTER(n, template_for_instance, neuron_type)                     \
    detail::register_direct_converter<nt::neuron_parameters<knp::neuron_traits::neuron_type>>(); \
    py::implicitly_convertible<core::Population<nt::neuron_type>, core::AllPopulationsVariant>();


inline void instance_populations_converters()
{
    BOOST_PP_SEQ_FOR_EACH(INSTANCE_POPULATION_CONVERTER, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_NEURONS))
}


void export_populations();
