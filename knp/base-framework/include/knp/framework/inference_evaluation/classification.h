/**
 * @file classification.h
 * @brief Evaluation of how good model performs by inference results
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
#include <knp/core/messaging/messaging.h>
#include <knp/framework/data_processing/image_classification.h>

#include <vector>


namespace knp::framework::inference_evaluation::classification
{

/**
 * @brief Processed inference result for single class.
 */
struct InferenceResultForClass
{
    /**
     * @brief Total amount of model predictions, equal to no_votes_ + correctly_predicted_ + incorrectly_predicted_.
     */
    size_t total_ = 0;
    /**
     * @brief Amount of model predictions when this class won, without any votes.
     */
    size_t no_votes_ = 0;
    /**
     * @brief Amount of correct model predictions.
     */
    size_t correctly_predicted_ = 0;
    /**
     * @brief Amount of incorrect model predictions.
     */
    size_t incorrectly_predicted_ = 0;
};


/**
 * @brief Process inference results. Suited for classification models.
 * @details Will process spikes data retrieved from inference, and return statistics containing how much times model
 * predicted some class correctly/incorrectly, predicted with no votes and total amount of votes.
 * @param spikes all spikes from inference.
 * @param dataset dataset.
 * @param classes_amount amount of classes in classification model.
 * @param steps_per_class amoount of steps required for one object, for example in MNIST you can translate image to
 * spikes that will be sent in 20 steps, so steps_per_class=20.
 * @return processed inference results for each class.
 */
[[nodiscard]] KNP_DECLSPEC std::vector<InferenceResultForClass> process_inference_results(
    const std::vector<knp::core::messaging::SpikeMessage> &spikes,
    const knp::framework::data_processing::classification::Dataset &dataset, size_t classes_amount,
    size_t steps_per_class);

/**
 * @brief Put inference results for each class to a stream in form of csv.
 * @param results_stream stream for output.
 * @param inference_results processed inference results.
 */
KNP_DECLSPEC void write_inference_results_to_stream_as_csv(
    std::ostream &results_stream, const std::vector<InferenceResultForClass> &inference_results);

}  //namespace knp::framework::inference_evaluation::classification
