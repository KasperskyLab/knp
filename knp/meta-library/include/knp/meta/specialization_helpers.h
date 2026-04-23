/**
 * @file specialization_helpers.h
 * @brief Some useful routines for working with template specializations.
 * @kaspersky_support Artiom N.
 * @date 11.08.2023
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

#include <memory>
#include <string>
#include <type_traits>
#ifndef _MSC_VER
#    include <cxxabi.h>
#endif

/**
 * @brief Metaprogramming library namespace.
 */
namespace knp::meta
{

/**
 * @brief Return type of variable as a string.
 * 
 * @tparam T type of variable.
 * 
 * @return name of a variable type.
 * 
 * @details See <a
 * href="https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c">Stack
 * overflow</a>.
 * 
 * @note Use as `type_name<decltype(variable)>()`.
 */
template <class T>
std::string type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void (*)(void*)> own(
#ifndef _MSC_VER
        abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
#else
        nullptr,
#endif
        std::free);
    std::string result = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value) result += " const";
    if (std::is_volatile<TR>::value) result += " volatile";
    if (std::is_lvalue_reference<T>::value)
        result += "&";
    else if (std::is_rvalue_reference<T>::value)
        result += "&&";
    return result;
}


/**
 * @class std::false_type
 * 
 * @brief The `integral_constant` instance to represent the `false` bool value.
 * 
 * @see <a href="https://en.cppreference.com/w/cpp/types/integral_constant">std::false_type description</a>
 */


/**
 * @brief Specialization test template.
 * 
 * @tparam T possible specialization of `Template`.
 * @tparam Template template.
 * 
 * @details See <a href="https://stackoverflow.com/questions/31762958/check-if-class-is-a-template-specialization">
 Stack Overflow</a>.
 * 
 * @code
    static_assert(is_specialization<std::vector<int>, std::vector>{}, "");
    static_assert(!is_specialization<std::vector<int>, std::list>{}, "");
 * @endcode
 */
template <class T, template <class...> class Template>
struct is_specialization : std::false_type
{
};


/**
 * @class std::true_type
 * 
 * @brief The `integral_constant` instance to represent the `true` bool value.
 * 
 * @see <a href="https://en.cppreference.com/w/cpp/types/integral_constant">std::true_type description</a>
 */


/**
 * @copydoc knp::meta::is_specialization
 */
template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type
{
};

}  // namespace knp::meta
