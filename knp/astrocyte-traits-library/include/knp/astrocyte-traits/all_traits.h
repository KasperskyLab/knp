/**
 * @file all_traits.h
 * @brief Registry of astrocyte models.
 * @author Yuliya Tsybina
 * @date 2026
 * @license Apache 2.0
 * @copyright © 2026 Lobachevsky University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <knp/meta/stringify.h>

#include <boost/mp11.hpp>

#include "astrocyte_simple.h"
#include "astrocyte_nonlinear.h"

/**
 * @brief Namespace for astrocyte traits.
 */
namespace knp::astrocyte_traits
{

/**
 * @brief Comma-separated list of astrocyte tags.
 */
#define ALL_ASTROCYTES AstrocyteSimple, AstrocyteNonlinear


/**
 * @brief List of astrocyte types.
 *
 * @details To add an astrocyte type to the list, define it in the ALL_ASTROCYTES macro.
 */
using AllAstrocytes = boost::mp11::mp_list<ALL_ASTROCYTES>;


/**
 * @brief A tuple that contains string names of astrocyte types.
 */
constexpr auto astrocytes_names = KNP_MAKE_TUPLE(ALL_ASTROCYTES);

}  // namespace knp::astrocyte_traits
