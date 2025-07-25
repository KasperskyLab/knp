/**
 * @file classification_dataset.h
 * @brief Definition of classification dataset
 * @kaspersky_support D. Postnikov
 * @date 21.07.2025
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

#include <knp/core/core.h>
#include <knp/core/impexp.h>
#include <knp/core/messaging/messaging.h>

#include <utility>
#include <vector>


namespace knp::framework::data_processing::classification
{

/**
 * @brief A struct that represents processed dataset.
 */
struct Dataset
{
    /**
     * @brief Vector of pairs of label and class data in spikes form, distributed in several steps.
     */
    std::vector<std::pair<unsigned, std::vector<bool>>> data_for_training_;

    /**
     * @brief Vector of pairs of label and class data in spikes form, distributed in several steps.
     */
    std::vector<std::pair<unsigned, std::vector<bool>>> data_for_inference_;

    /**
     * @brief Amount of steps the converted class data will be sent.
     */
    size_t steps_per_class_;

    /**
     * @brief Amount of steps required for training.
     */
    size_t steps_required_for_training_;

    /**
     * @brief Amount of steps required for inference.
     */
    size_t steps_required_for_inference_;
};

}  // namespace knp::framework::data_processing::classification
