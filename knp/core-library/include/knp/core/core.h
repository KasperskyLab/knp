/**
 * @file core.h
 * @brief Class definition for core library basic entities.
 * @kaspersky_support Artiom N.
 * @date 11.01.2023
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

#include <knp/core/tag.h>
#include <knp/core/uid.h>


/**
 * @namespace knp
 * 
 * @brief General framework namespace.
 */


/**
 * @brief Core library namespace.
 */
namespace knp::core
{
/**
 * @brief Common parameters for several different entities.
 * 
 * @details For example, BaseData is used in entities of Backend, Device, Population and Projection classes.
 */
struct BaseData
{
    /**
     * @brief Entity unique identifier.
     */
    UID uid_;
    /**
     * @brief Entity tags.
     */
    TagMap tags_;
}
#if defined(__GNUC__) || defined(__clang__)
/**
 * @brief Attribute for alignment.
 * 
 * @return none.
 */
__attribute__((aligned(sizeof(size_t) * 8)));
#else
;  // NOLINT
#endif


/**
 * @brief Type used to store a step number in the form of a 64-bit unsigned integer.
 */
using Step = uint64_t;

}  // namespace knp::core
