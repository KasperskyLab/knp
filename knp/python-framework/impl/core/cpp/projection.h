/**
 * @file projection.h
 * @brief Python bindings header for Projection.
 * @kaspersky_support Artiom N.
 * @date 16.02.2024
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

#include <knp/core/projection.h>
#include <knp/synapse-traits/all_traits.h>

#include <iostream>
#include <memory>
#include <optional>
#include <tuple>

#include <boost/python/tuple.hpp>

#include "common.h"
#include "tuple_converter.h"


template <typename ElemParametersType>
struct ProjectionGeneratorProxy
{
    explicit ProjectionGeneratorProxy(const py::object &gen_func) : gen_func_(gen_func)
    {
        if (!PyCallable_Check(gen_func.ptr()))
        {
            PyErr_SetString(PyExc_TypeError, "Passed generator is not callable.");
            py::throw_error_already_set();
        }
    }

    std::optional<typename core::Projection<ElemParametersType>::Synapse> operator()(size_t index)
    {
        auto res = py::call<py::object>(gen_func_.ptr(), index);
        if (res.is_none())
        {
            return std::nullopt;
        }

        return py::extract<typename core::Projection<ElemParametersType>::Synapse>(res);
    }

    const py::object &gen_func_;
};


template <typename ElemType>
std::shared_ptr<core::Projection<ElemType>> projection_constructor_wrapper(
    core::UID uid, core::UID presynaptic_uid, core::UID postsynaptic_uid, const py::object &gen_func,
    size_t num_iterations)
{
    return std::make_shared<core::Projection<ElemType>>(
        uid, presynaptic_uid, postsynaptic_uid, ProjectionGeneratorProxy<ElemType>(gen_func), num_iterations);
};


template <typename ElemType>
std::shared_ptr<core::Projection<ElemType>> projection_constructor_wrapper(
    core::UID presynaptic_uid, core::UID postsynaptic_uid, const py::object &gen_func, size_t num_iterations)
{
    return std::make_shared<core::Projection<ElemType>>(
        presynaptic_uid, postsynaptic_uid, ProjectionGeneratorProxy<ElemType>(gen_func), num_iterations);
};


template <typename ElemType>
void projection_synapses_add_wrapper(
    core::Projection<ElemType> &projection, const py::object &gen_func, size_t num_iterations)
{
    projection.add_synapses(ProjectionGeneratorProxy<ElemType>(gen_func), num_iterations);
};


namespace st = knp::synapse_traits;

#define INSTANCE_PROJECTION_CONVERTER(n, template_for_instance, synapse_type)                       \
    detail::register_direct_converter<st::synapse_parameters<knp::synapse_traits::synapse_type>>(); \
    py::implicitly_convertible<core::Projection<st::synapse_type>, core::AllProjectionsVariant>();


inline void instance_projections_converters()
{
    BOOST_PP_SEQ_FOR_EACH(INSTANCE_PROJECTION_CONVERTER, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_SYNAPSES))
}

void instance_projections_converters();
void export_projections();
