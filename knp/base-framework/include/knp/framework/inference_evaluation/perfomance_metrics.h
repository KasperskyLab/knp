/**
 * @file perfomance_metrics.h
 * @brief Header file for performance metrics functions.
 * @kaspersky_support D. Postnikov
 * @date 24.07.2025
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
#include <knp/core/messaging/messaging.h>


/**
 * @brief Namespace for inference evaluation.
 */
namespace knp::framework::inference_evaluation
{

/**
 * @brief Calculate model precision.
 * @details Precision is the ratio of true positives to the sum of true positives and false positives.
 * It measures the proportion of correct predictions among all positive predictions made by the model.
 * @param true_positives number of true positives.
 * @param false_positives number of false positives.
 * @return model precision.
 */
KNP_DECLSPEC float get_precision(size_t true_positives, size_t false_positives);


/**
 * @brief Calculate model recall.
 * @details Recall is the ratio of true positives to the sum of true positives and false negatives.
 * It measures the proportion of correct predictions among all actual positive instances.
 * @param true_positives number of true positives.
 * @param false_negatives number of false negatives.
 * @return model recall.
 */
KNP_DECLSPEC float get_recall(size_t true_positives, size_t false_negatives);


/**
 * @brief Calculate model prevalence.
 * @details Prevalence is the ratio of the sum of true positives and false negatives to the total number of instances.
 * It measures the proportion of actual positive instances in the dataset.
 * @param true_positives number of true positives.
 * @param false_negatives number of false negatives.
 * @param false_positives number of false positive.
 * @param true_negatives number of true negatives.
 * @return model prevalence.
 */
KNP_DECLSPEC float get_prevalence(
    size_t true_positives, size_t false_negatives, size_t false_positives, size_t true_negatives);


/**
 * @brief Calculate model accuracy.
 * @param true_positives number of true positives.
 * @param false_negatives number of false negatives.
 * @param false_positives number of false positive.
 * @param true_negatives number of true negatives.
 * @return model accuracy.
 */
KNP_DECLSPEC float get_accuracy(
    size_t true_positives, size_t false_negatives, size_t false_positives, size_t true_negatives);


/**
 * @brief Calculate model F-score
 * @details F-score is the harmonic mean of precision and recall. It measures the balance between precision and recall.
 * @param precision model precision.
 * @param recall model recall.
 * @return model F-score.
 */
KNP_DECLSPEC float get_f_score(float precision, float recall);

}  // namespace knp::framework::inference_evaluation
