/**
 * @file perfomance_metrics.h
 * @brief Evaluation of how good model performs by inference results
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

#include <knp/core/messaging/messaging.h>


namespace knp::framework::inference_evaluation
{

/**
 * @brief Calculate precision of model.
 * @param correct_predictions Correct predictions.
 * @param incorrect_predictions Incorrect predictions.
 */
float get_precision(size_t correct_predictions, size_t incorrect_predictions);


/**
 * @brief Calculate recall of model.
 * @param correct_predictions Correct predictions.
 * @param incorrect_predictions Incorrect predictions.
 */
float get_recall(size_t correct_predictions, size_t incorrect_predictions);


/**
 * @brief Calculate f measure of model.
 * @param precision Model's precision.
 * @param recall Model's recall.
 */
float get_f_measure(float precision, float recall);

}  //namespace knp::framework::inference_evaluation
