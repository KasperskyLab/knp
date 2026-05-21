/**
 * @file impexp.h
 * @brief Header for import and export control.
 * @kaspersky_support Artiom N.
 * @date 25.10.2021
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
 * @name _KNP_SYMBOL_IMPORT
 * 
 * @brief Symbols import macro.
 */

/**
 * @name _KNP_SYMBOL_EXPORT
 * 
 * @brief Symbols export macro.
 */

/**
 * @name _KNP_SYMBOL_LOCAL
 * 
 * @brief Local symbol.
 */

/**
 * @name _KNP_SYMBOL_EXTERN
 * 
 * @brief External symbol.
 */

#if defined(_MSC_VER)
#    define _KNP_SYMBOL_EXPORT __declspec(dllexport)
#    define _KNP_SYMBOL_IMPORT __declspec(dllimport)
#    define _KNP_SYMBOL_EXTERN extern
#    define _KNP_SYMBOL_LOCAL
#elif defined(_WIN32) && defined(__CYGWIN__)
#    define _KNP_SYMBOL_EXPORT __attribute__((dllexport))
#    define _KNP_SYMBOL_IMPORT __attribute__((dllimport))
#    define _KNP_SYMBOL_EXTERN extern
#    define _KNP_SYMBOL_LOCAL
#elif defined(__GNUC__) || defined(__clang__)
#    define _KNP_SYMBOL_IMPORT __attribute__((visibility("default")))
#    define _KNP_SYMBOL_EXPORT __attribute__((visibility("default")))
#    define _KNP_SYMBOL_EXTERN
#    define _KNP_SYMBOL_LOCAL __attribute__((visibility("hidden")))
#else
#    define _KNP_SYMBOL_IMPORT
#    define _KNP_SYMBOL_EXPORT
#    define _KNP_SYMBOL_EXTERN
#    define _KNP_SYMBOL_LOCAL
#endif

// Defined if Kaspersky Neuromorphic Platform is built as a shared library.
#if defined(_KNP_BUILD_SHARED_LIBS)
#    define KNP_EXPORT _KNP_SYMBOL_EXPORT
#    define KNP_IMPORT _KNP_SYMBOL_IMPORT
#    define KNP_EXTERN _KNP_SYMBOL_EXTERN
#    define KNP_LOCAL _KNP_SYMBOL_LOCAL
#else
#    define KNP_EXPORT
#    define KNP_IMPORT
#    define KNP_EXTERN
#    define KNP_LOCAL
#endif

#if defined(_KNP_INTERNAL)
#    define KNP_DECLSPEC KNP_EXPORT
#else
#    define KNP_DECLSPEC KNP_IMPORT
#endif
