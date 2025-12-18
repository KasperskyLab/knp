/**
 * @file processor.h
 * @brief Header file for processing inference results.
 * @kaspersky_support D. Postnikov
 * @date 05.09.2025
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
#include <knp/framework/data_processing/classification/image.h>

#include <vector>

#include "result.h"


/**
 * @brief Namespace for inference evaluation.
 */
namespace knp::framework::inference_evaluation
{

/**
 * @brief Namespace for classification model inference evaluation.
 */
namespace classification
{

/**
 * @details The `InferenceResultsProcessor` class is a definition of an interface used to process 
 * inference results, suited for classification models.
 */
class KNP_DECLSPEC InferenceResultsProcessor
{
public:
    /**
     * @brief Process inference results. 
     * @param spikes all spikes from inference.
     * @param dataset dataset used for inference.
     * @details The method processes spikes and updates internal state of the processor. It then 
     * calculates the performance metrics for each class in the dataset. 
     */
    void process_inference_results(
        const std::vector<knp::core::messaging::SpikeMessage> &spikes,
        const knp::framework::data_processing::classification::Dataset &dataset);

    /**
     * @brief Write inference results to a stream in CSV format.
     * @param results_stream stream to write the results to.
     * @details The method writes processed inference results to the specified stream in CSV format.
     * The results include true positives, false negatives, false positives, and true negatives for each class,
     * as well as the calculated precision, recall, prevalence, accuracy, and F-score.
     */
    void write_inference_results_to_stream_as_csv(std::ostream &results_stream);

    /**
     * @brief Get processed inference results.
     * @return vector of `InferenceResult` objects, each representing results for a single class.
     */
    [[nodiscard]] const std::vector<InferenceResult> &get_inference_results() const { return inference_results_; }

private:
    /**
     * @brief Processed inference results.
     */
    std::vector<InferenceResult> inference_results_;
};

}  // namespace classification

}  // namespace knp::framework::inference_evaluation
