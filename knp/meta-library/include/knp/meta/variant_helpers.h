/**
 * @file variant_helpers.h
 * @brief Some useful routines for working with `std::variant`.
 * @kaspersky_support Artiom N.
 * @date 09.08.2023
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

#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <boost/mp11.hpp>


/**
 * @brief Metaprogramming library namespace.
 */
namespace knp::meta
{

/**
 * @brief Load elements from one container of all variants to another container that contains a subset of all
 * variants.
 * @tparam SupportedTypes subset of variants.
 * @tparam AllVariants all supported variants.
 * @tparam ToContainer target container.
 * @param from_container source container.
 * @param to_container target container.
 */
template <typename SupportedTypes, typename AllVariants, typename ToContainer>
void load_from_container(const std::vector<AllVariants> &from_container, ToContainer &to_container)
{
    to_container.clear();
    to_container.reserve(from_container.size());

    for (const auto &p : from_container)
    {
        std::visit(
            [&to_container](auto &arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (boost::mp11::mp_find<SupportedTypes, T>{} != boost::mp11::mp_size<SupportedTypes>{})
                {
                    to_container.push_back(typename ToContainer::value_type{arg});
                }
                else
                {
                    throw std::runtime_error("Type is not supported.");
                }
            },
            p);
    }
}


/**
 * @brief Convert from one set of arguments to another.
 * @note This is is a helper structure. Use `variant_cast` instead.
 * @tparam FromArgs source variant arguments.
 */
template <class... FromArgs>
struct variant_cast_proxy
{
    /**
     * @brief Buffer variable.
     */
    std::variant<FromArgs...> v;
    /**
     * @brief Cast operator.
     * @tparam ToArgs target variant parameters.
     * @return Same value as source, cast to a different variant.
     */
    template <class... ToArgs>
    operator std::variant<ToArgs...>() const
    {
        return std::visit([](auto &&arg) -> std::variant<ToArgs...> { return arg; }, v);
    }
};


/**
 * @brief Cast from one variant type to another.
 * @tparam Args source variant arguments
 * @param v source value.
 * @return Source value cast to `variant_cast_proxy`. `variant_cast_proxy` then implicitly converts the value to target
 * variant.
 */
template <class... Args>
static auto variant_cast(const std::variant<Args...> &v) -> variant_cast_proxy<Args...>
{
    return {v};
}

}  // namespace knp::meta
