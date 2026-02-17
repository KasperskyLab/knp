/**
 * @file logging.h
 * @brief Global logging settings.
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

#include <string>


/**
 * @brief Logging namespace.
 */
namespace knp::framework::logging
{

/**
 * @brief Levels of logging. Each level includes all levels below it including itself. So for example "warn" will
 * enable "warn","error","critical", "none".
 * @note This enum have direct mapping to spdlog's logging levels.
 *
 */
enum Level : int
{
    trace,
    debug,
    info,
    warn,
    error,
    critical,
    none
};


/**
 * @brief Set level of logging.
 * @param level Logging level.
 */
void set_level(Level level);


/**
 * @brief Get level of logging.
 * @return Logging level.
 */
Level get_level();


/**
 * @brief Convert level to string.
 * @param level Logging level.
 * @return Converted string.
 */
std::string level_to_str(Level level);


/**
 * @brief Convert string to level.
 * @param str Level string.
 * @return Converted logging level.
 */
Level str_to_level(std::string_view str);

}  //namespace knp::framework::logging
