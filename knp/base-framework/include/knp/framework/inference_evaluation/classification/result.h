/**
 * @file result.h
 * @brief Structure to hold inference results.
 * @kaspersky_support D. Postnikov
 * @date 16.07.2025
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

#pragma once

#include <knp/core/impexp.h>

namespace knp::framework::inference_evaluation::classification
{

/**
 * @brief Processed inference result for single class.
 */
struct KNP_DECLSPEC InferenceResult
{
    /**
     * @brief Amount of times model, that is supposed  to predict dog, predicted dog when it is a dog.
     */
    size_t true_positives_ = 0;

    /**
     * @brief Amount of times model, that is supposed to predict dog, predicted not a dog when it is a dog.
     */
    size_t false_negatives_ = 0;

    /**
     * @brief Amount of times model, that is supposed to predict dog, predicted dog when it is not a dog.
     */
    size_t false_positives_ = 0;

    /**
     * @brief Amount of times model, that is supposed to predict dog, predicted not a dog when it is a not a dog.
     */
    size_t true_negatives_ = 0;

    /**
     * @brief Shortcut for getting total votes.
     * @return Total votes.
     */
    [[nodiscard]] size_t get_total_votes() const { return true_positives_ + false_negatives_ + false_positives_; }
};

}  // namespace knp::framework::inference_evaluation::classification
