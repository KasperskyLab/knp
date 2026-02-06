/**
 * @file resource_from_weight.h
 * @brief Function for calculating resource from weight.
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

#include <stdexcept>
#include <string>
#include <utility>


inline float resource_from_weight(float weight, float min_weight, float max_weight)
{
    // Max weight is only possible with infinite resource, so we should select a value less than that.
    float eps = 1e-6;
    if (min_weight > max_weight) std::swap(min_weight, max_weight);
    if (weight < min_weight || weight >= max_weight - eps)
        throw std::logic_error(
            std::string("Weight should not be less than") + std::to_string(min_weight) + ", more than" +
            std::to_string(max_weight) + "or too close to it. Weight = " + std::to_string(weight));
    double diff = max_weight - min_weight;
    double over = weight - min_weight;
    return static_cast<float>(over * diff / (diff - over));
}
