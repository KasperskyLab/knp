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
 * @brief Tag name used to store an object's human‑readable name.
 *
 * @details The literal `name` is the key under which a string tag is placed in the object's @ref TagMap. It is used 
 * by the helper functions below.
 */
constexpr char name_tag[]{"name"};


/**
 * @brief Retrieve the object tag value by name.
 *
 * @tparam Type object type.
 *
 * @param object object from which the name is requested.
 *
 * @return tag value if the tag name is present in the object; otherwise the object's UID converted to a string.
 *
 * @details The function checks whether the object's tag map contains @ref name_tag. If it does, the stored tag value 
 * is returned. When the tag is absent, the object's unique identifier is used as a fallback.
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
 * @brief Assign a tag value to an object's tag.
 *
 * @tparam Type object type.
 *
 * @param object object whose tag value will be set.
 * @param name value to assign.
 *
 * @details The function overwrites any existing value associated with @ref name_tag and stores a copy of @p name 
 * as a string in the object's @ref TagMap.
 */
template <typename Type>
void set_name(Type& object, std::string_view name)
{
    object.get_tags()[name_tag] = std::string(name);
}

}  //namespace knp::framework::tags
