/**
 * @file input_interpreters.h
 * @brief Header for input interpreter.
 * @kaspersky_support Vartenkov Andrey
 * @date 31.06.2023
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


/**
 * @brief Input channel namespace.
 */
namespace knp::framework::io::input
{
/**
 * @brief Determine if an input value is a spike and convert an input value to boolean.
 * 
 * @tparam ValueType input value type (usually an arithmetic type) that can be cast to bool implicitly.
 * 
 * @param val input value.
 * 
 * @return `true` if an input value is a spike, `false` if an input value is not a spike.
 */
template <class ValueType>
bool interpret_as_bool(ValueType val)
{
    return val;
}


/**
 * @brief Interpret an input value as a spike if an input value is equal or greater than a threshold value.
 * 
 * @tparam ValueType input value type that should have a defined operator "less than".
 * 
 * @param threshold threshold value.
 * 
 * @return boolean function that determines if a value causes a spike or not.
 */
template <class ValueType>
auto interpret_with_threshold(ValueType threshold)
{
    // We're using !(a < b) instead of (a >= b) because of the values which only have operator "less than".
    return [threshold](ValueType val) -> bool { return !(val < threshold); };
}

}  // namespace knp::framework::io::input
