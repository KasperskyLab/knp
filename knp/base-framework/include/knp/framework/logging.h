/**
 * @file logging.h
 * @brief Global logging API settings.
 * @kaspersky_support Postnikov D.
 * @date 17.02.2026
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

#include <knp/core/impexp.h>

#include <string>


/**
 * @brief Logging utilities for the KNP framework.
 *
 * @details The namespace provides a thin wrapper around *spdlog* that maps the framework‑specific `Level` enum 
 * to spdlog’s logging levels and offers convenient conversion functions.
 */
namespace knp::framework::logging
{


/**
 * @brief Logging levels.
 *
 * @details Each level implicitly enables all levels that are numerically lower. For example, setting the level to 
 * `warn` enables logging for `warn`, `error`, `critical` and `none`.  The values correspond directly to spdlog’s 
 * levels. The special value `none` disables logging completely.
 *
 * @note This enum has a one‑to‑one mapping to spdlog’s logging levels.
 */
enum Level : int
{
    /**
     * @brief Trace‑level messages.
     */
    trace,
    /**
     * @brief Debug-level messages.
     */
    debug,
    /**
     * @brief Informational messages.
     */
    info,
    /**
     * @brief Warning messages.
     */
    warn,
    /**
     * @brief Error messages.
     */
    error,
    /**
     * @brief Critical error messages.
     */
    critical,
    /**
     * @brief Logging disabled.
     */
    none
};


/**
 * @brief Set the global logging level.
 *
 * @param level logging level to apply.
 */
KNP_DECLSPEC void set_level(Level level);


/**
 * @brief Retrieve the current global logging level.
 *
 * @return current logging level.
 */
KNP_DECLSPEC Level get_level();


/**
 * @brief Convert a logging level to its string representation.
 *
 * @param level logging level to convert.
 *
 * @return string that represents the given level; returns `none` for the @ref Level::none level.
 */
KNP_DECLSPEC std::string level_to_str(Level level);


/**
 * @brief Convert a string to a logging level.
 *
 * @param str string that specifies a logging level.
 *
 * @return logging level corresponding to the string; returns `none` if the string is empty, 
 * equals `none`, or cannot be parsed.
 */
KNP_DECLSPEC Level str_to_level(std::string_view str);

}  //namespace knp::framework::logging
