/**
 * @file common.h
 * @brief Common header for Python bindings.
 * @kaspersky_support Artiom N.
 * @date 05.02.2024
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
#include <knp/core/device.h>
#include <knp/core/message_bus.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/message_envelope.h>
#include <knp/core/messaging/message_header.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/messaging/synaptic_impact_message.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/core/subscription.h>
#include <knp/core/uid.h>

#if defined(__GNUC__) && (__GNUC__ >= 14)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdangling-reference"
#    include <spdlog/spdlog.h>
#    pragma GCC diagnostic pop
#elif defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wunknown-warning-option"
#    pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#    include <spdlog/spdlog.h>
#    pragma clang diagnostic pop
#else
#    include <spdlog/spdlog.h>
#endif

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <boost/python.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>


namespace py = boost::python;
namespace core = knp::core;

namespace detail
{
template <typename F>
struct function_traits : public function_traits<decltype(&F::operator())>
{
};


template <typename R, typename C, typename... Args>
struct function_traits<R (C::*)(Args...) const>
{
    using function_type = std::function<R(Args...)>;
};
}  // namespace detail


template <typename F>
using function_type_t = typename detail::function_traits<F>::function_type;

template <typename F>
function_type_t<F> to_function(F lambda)
{
    return static_cast<function_type_t<F>>(lambda);
}


template <typename R, typename... Args>
py::object make_handler(std::function<R(Args...)> func)
{
    return py::detail::make_function_aux(
        func, py::default_call_policies(), boost::mpl::vector<std::decay_t<R>, Args...>());
}


template <typename F>
py::object make_handler(F func)
{
    return make_handler(to_function(func));
}


template <typename T>
std::vector<T> py_iterable_to_vector(const py::object &iterable)
{
    return std::vector<T>(py::stl_input_iterator<T>(iterable), py::stl_input_iterator<T>());
}


// Adapter a member function that returns a unique_ptr to
// a python function object that returns a raw pointer but
// explicitly passes ownership to Python.
// https://stackoverflow.com/questions/20581679/boost-python-how-to-expose-stdunique-ptr .
template <typename T, typename C, typename... Args>
boost::python::object adapt_unique(std::unique_ptr<T> (C::*fn)(Args...))
{
    return boost::python::make_function(
        [fn](C &self, Args... args) { return (self.*fn)(args...).release(); },
        boost::python::return_value_policy<boost::python::manage_new_object>(),
        boost::mpl::vector<T *, C &, Args...>());
}

std::string get_py_class_name(const py::object &obj_class);
