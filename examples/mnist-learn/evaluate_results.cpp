/**
 * @file evaluate_results.cpp
 * @brief Function for evaluating inference results.
 * @kaspersky_support D. Postnikov
 * @date 04.02.2026
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

#include "evaluate_results.h"

#include <knp/framework/inference_evaluation/classification/processor.h>

#include <vector>


void evaluate_results(const std::vector<knp::core::messaging::SpikeMessage>& inference_spikes, const Dataset& dataset)
{
    // Online Help link: https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=306417
    knp::framework::inference_evaluation::classification::InferenceResultsProcessor inference_processor;
    inference_processor.process_inference_results(inference_spikes, dataset);

    inference_processor.write_inference_results_to_stream_as_csv(std::cout);
}
