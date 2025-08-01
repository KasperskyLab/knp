/**
 * @file perfomance_metrics.h
 * @brief Functions to calculate model statistics.
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


namespace knp::framework::inference_evaluation
{

/**
 * @brief Calculate precision.
 * @param true_positives Amount of times model, that is supposed to predict dog, predicted dog when it is a dog.
 * @param false_positives Amount of times model, that is supposed to predict dog, predicted dog when it is not a dog.
 */
KNP_DECLSPEC float get_precision(size_t true_positives, size_t false_positives);


/**
 * @brief Calculate recall.
 * @param true_positives Amount of times model, that is supposed to predict dog, predicted dog when it is a dog.
 * @param false_negatives Amount of times model, that is supposed to predict dog, predicted not a dog when it is a dog.
 */
KNP_DECLSPEC float get_recall(size_t true_positives, size_t false_negatives);


/**
 * @brief Calculate prevalence.
 * @param true_positives Amount of times model, that is supposed  to predict dog, predicted dog when it is a dog.
 * @param false_negatives Amount of times model, that is supposed to predict dog, predicted not a dog when it is a dog.
 * @param false_positives Amount of times model, that is supposed to predict dog, predicted dog when it is not a dog.
 * @param true_negatives Amount of times model, that is supposed to predict dog, predicted not a dog when it is a not a
 * dog.
 */
KNP_DECLSPEC float get_prevalence(
    size_t true_positives, size_t false_negatives, size_t false_positives, size_t true_negatives);


/**
 * @brief Calculate accuracy.
 * @param true_positives Amount of times model, that is supposed  to predict dog, predicted dog when it is a dog.
 * @param false_negatives Amount of times model, that is supposed to predict dog, predicted not a dog when it is a dog.
 * @param false_positives Amount of times model, that is supposed to predict dog, predicted dog when it is not a dog.
 * @param true_negatives Amount of times model, that is supposed to predict dog, predicted not a dog when it is a not a
 * dog.
 */
KNP_DECLSPEC float get_accuracy(
    size_t true_positives, size_t false_negatives, size_t false_positives, size_t true_negatives);


/**
 * @brief Calculate f measure.
 * @param precision Precision.
 * @param recall Recall.
 */
KNP_DECLSPEC float get_f_measure(float precision, float recall);

}  //namespace knp::framework::inference_evaluation
