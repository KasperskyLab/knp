/**
 * @file parse_arguments.h
 * @brief Parsing of command line arguments.
 * @kaspersky_support D. Postnikov
 * @date 03.02.2026
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

#include <optional>

#include "model_desc.h"


/**
 * @brief Parse command-line arguments and configure model parameters.
 * 
 * @details This function processes command-line arguments, validates the configuration, and populates the @ref ModelDescription 
 * structure. 
 * 
 * @param argc number of command-line arguments.
 * @param argv array of command-line argument strings.
 * @return @ref ModelDescription containing parsed configuration parameters; nothing if parsing failed, help was requested, 
 * or required parameters are missing.
 */
std::optional<ModelDescription> parse_arguments(int argc, char** argv);
