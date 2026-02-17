/**
 * @file logging.cpp
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

#include <knp/framework/logging.h>

#include <spdlog/spdlog.h>


namespace knp::framework::logging
{

spdlog::level::level_enum convert_level_to_spdlog_level(Level level)
{
    auto spdlog_level = static_cast<spdlog::level::level_enum>(level);
    if (spdlog_level >= spdlog::level::level_enum::n_levels)
    {
        SPDLOG_ERROR("Could not convert logging level to spdlog's logging level. Returning level \"none\".");
        spdlog_level = spdlog::level::off;
    }
    return spdlog_level;
}


Level convert_spdlog_level_to_level(spdlog::level::level_enum spdlog_level)
{
    auto level = static_cast<Level>(spdlog_level);
    if (level > none)
    {
        SPDLOG_ERROR("Could not convert spdlog's logging level to knp's logging level. Returning level \"none\".");
        level = none;
    }
    return level;
}


void set_level(Level level)
{
    spdlog::set_level(convert_level_to_spdlog_level(level));
}


Level get_level()
{
    return convert_spdlog_level_to_level(spdlog::get_level());
}


std::string level_to_str(Level level)
{
    if (level == none) return "none";
    return spdlog::level::to_string_view(convert_level_to_spdlog_level(level)).begin();
}


Level str_to_level(std::string_view str)
{
    if (str == "none") return none;
    auto level = spdlog::level::from_str(str.begin());
    if (level == spdlog::level::off)
    {
        SPDLOG_ERROR("Could not convert string to level.");
        return none;
    }
    return convert_spdlog_level_to_level(level);
}

}  //namespace knp::framework::logging
