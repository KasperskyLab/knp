/**
 * @file name.h
 * @brief Functions for working with name tag.
 * @kaspersky_support Postnikov D.
 * @date 03.04.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include <string>


/**
 * @brief Tags namespace.
 */
namespace knp::framework::tags
{

/**
 * @brief Tag name for setting names.
 */
constexpr char name_tag[]{"name"};


/**
 * @brief Get name tag.
 * @tparam Type Object type.
 * @param object Object from which we want to retrieve name.
 * @return If name was specified in object, it is returned, otherwise UID is returned.
 *
 */
template <typename Type>
[[nodiscard]] std::string get_name(const Type& object)
{
    if (object.get_tags().exists(name_tag))
    {
        return object.get_tags().template get_tag<std::string>(name_tag);
    }
    return std::string(object.get_uid());
}


/**
 * @brief Set name tag.
 * @tparam Type Object type.
 * @param object Object in which we want to set name.
 * @param name Name.
 */
template <typename Type>
void set_name(Type& object, std::string_view name)
{
    object.get_tags()[name_tag] = std::string(name);
}

}  //namespace knp::framework::tags
