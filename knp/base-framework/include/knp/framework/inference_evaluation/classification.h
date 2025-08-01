/**
 * @file classification.h
 * @brief Evaluation of how good model performs by inference results.
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
#include <knp/framework/data_processing/classification/image.h>

#include <vector>

#include "perfomance_metrics.h"


namespace knp::framework::inference_evaluation::classification
{

/**
 * @brief Processed inference result for single class.
 */
class KNP_DECLSPEC InferenceResultForClass
{
public:
    /**
     * @brief Get true positives.
     * @ret Amount of times model, that is supposed  to predict dog, predicted dog when it is a dog.
     */
    [[nodiscard]] size_t get_true_positives() const { return true_positives_; }

    /**
     * @brief Get false negatives.
     * @ret Amount of times model, that is supposed to predict dog, predicted not a dog when it is a dog.
     */
    [[nodiscard]] size_t get_false_negatives() const { return false_negatives_; }

    /**
     * @brief Get false positives.
     * @ret Amount of times model, that is supposed to predict dog, predicted dog when it is not a dog.
     */
    [[nodiscard]] size_t get_false_positives() const { return false_positives_; }

    /**
     * @brief Get true negatives.
     * @ret Amount of times model, that is supposed to predict dog, predicted not a dog when it is a not a dog.
     */
    [[nodiscard]] size_t get_true_negatives() const { return true_negatives_; }

    /**
     * @brief Shortcut for getting total votes.
     * @ret Total votes.
     */
    [[nodiscard]] size_t get_total_votes() const { return true_positives_ + false_negatives_ + false_positives_; }

    /**
     * @detail A class to process inference results.
     */
    class KNP_DECLSPEC InferenceResultsProcessor
    {
    public:
        /**
         * @brief Process inference results. Suited for classification models.
         * @param spikes All spikes from inference.
         * @param dataset Dataset.
         * @return processed inference results for each class.
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
         * @ret Inference results.
         */
        [[nodiscard]] auto const &get_inference_results() const { return inference_results_; }

    private:
        /**
         * @brief Processed inference results.
         */
        std::vector<InferenceResultForClass> inference_results_;

        /**
         * @brief An internal class to help with evaluation.
         */
        class EvaluationHelper;
    };

private:
    size_t true_positives_ = 0, false_negatives_ = 0, false_positives_ = 0, true_negatives_ = 0;
};

}  //namespace knp::framework::inference_evaluation::classification
