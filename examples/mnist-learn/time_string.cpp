/**
 * @file time_string.cpp
 * @brief Functions for network training.
 * @kaspersky_support D. Postnikov
 * @date 28.03.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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

#include "time_string.h"

#include <chrono>


std::string get_time_string()
{
    auto time_now = std::chrono::system_clock::now();
    std::time_t c_time = std::chrono::system_clock::to_time_t(time_now);
    std::string result(std::ctime(&c_time));
    return result;
}
