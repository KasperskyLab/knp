/**
 * @file processor.h
 * @brief Processing inference results.
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


namespace knp::framework::inference_evaluation::classification
{

/**
 * @details A class to process inference results.
 */
class KNP_DECLSPEC InferenceResultsProcessor
{
public:
    /**
     * @brief Process inference results. Suited for classification models.
     * @param spikes All spikes from inference.
     * @param dataset Dataset.
     */
    void process_inference_results(
        const std::vector<knp::core::messaging::SpikeMessage> &spikes,
        const knp::framework::data_processing::classification::Dataset &dataset);

    /**
     * @brief Put inference results for each class to a stream in form of csv.
     * @param results_stream stream for output.
     */
    void write_inference_results_to_stream_as_csv(std::ostream &results_stream);

    /**
     * @brief Get inference results.
     * @return Inference results.
     */
    [[nodiscard]] const std::vector<InferenceResult> &get_inference_results() const { return inference_results_; }

private:
    /**
     * @brief Processed inference results.
     */
    std::vector<InferenceResult> inference_results_;
};

}  // namespace knp::framework::inference_evaluation::classification
