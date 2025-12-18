/**
 * @file result.h
 * @brief Header file for inference results.
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


/**
 * @brief Namespace for classification model inference evaluation.
 */
namespace knp::framework::inference_evaluation::classification
{

/**
 * @brief The `InferenceResult` structure represents inference results for a single class.
 */
struct KNP_DECLSPEC InferenceResult
{
    /**
     * @brief Number of true positives for the class.
     * @details A true positive occurs when the model correctly predicts the class.
     */
    size_t true_positives_ = 0;

    /**
     * @brief Number of false negatives for the class.
     * @details A false negative occurs when the model fails to predict the class when it is present.
     */
    size_t false_negatives_ = 0;

    /**
     * @brief Number of false positives for the class.
     * @details A false positive occurs when the model predicts the class when it is not present.
     */
    size_t false_positives_ = 0;

    /**
     * @brief Number of true negatives for the class.
     * @details A true negative occurs when the model correctly fails to predict the class when it is not present.
     */
    size_t true_negatives_ = 0;

    /**
     * @brief Get total votes for the class.
     * @return total votes for the class, which is the sum of true positives, false negatives, false positives, and true negatives.
     */
    [[nodiscard]] size_t get_total_votes() const { return true_positives_ + false_negatives_ + false_positives_; }
};

}  // namespace knp::framework::inference_evaluation::classification
