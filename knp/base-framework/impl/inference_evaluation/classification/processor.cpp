/**
 * @file classification.cpp
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

#include <knp/core/messaging/messaging.h>
#include <knp/framework/inference_evaluation/classification/processor.h>
#include <knp/framework/inference_evaluation/perfomance_metrics.h>

#include <algorithm>
#include <utility>


namespace knp::framework::inference_evaluation::classification
{

class EvaluationHelper
{
public:
    explicit EvaluationHelper(
        const knp::framework::data_processing::classification::Dataset &dataset,
        std::vector<InferenceResult> &inference_results);

    void process_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices, size_t step);

    [[nodiscard]] std::vector<InferenceResult> process_inference_predictions() const;

private:
    struct Prediction
    {
        const size_t predicted_class_;
        const size_t votes_;
    };

    // All predictions of model.
    std::vector<Prediction> predictions_;

    // Votes for some class each steps_per_class_ steps.
    std::vector<size_t> class_votes_;

    const knp::framework::data_processing::classification::Dataset &dataset_;
    std::vector<InferenceResult> &inference_results_;
};


EvaluationHelper::EvaluationHelper(
    const knp::framework::data_processing::classification::Dataset &dataset,
    std::vector<InferenceResult> &inference_results)
    : class_votes_(dataset.get_amount_of_classes(), 0), dataset_(dataset), inference_results_(inference_results)
{
}


void EvaluationHelper::process_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices, size_t step)
{
    for (auto i : firing_neuron_indices) ++class_votes_[i % dataset_.get_amount_of_classes()];
    if (!((step + 1) % dataset_.get_steps_per_frame()))
    {
        size_t n_max = 0;
        size_t predicted_state = 0;

        for (size_t i = 0; i < dataset_.get_amount_of_classes(); ++i)
        {
            if (class_votes_[i] > n_max)
            {
                n_max = class_votes_[i];
                predicted_state = i;
            }
        }
        predictions_.emplace_back(Prediction{predicted_state, n_max});
        std::fill(class_votes_.begin(), class_votes_.end(), 0);
    }
}


std::vector<InferenceResult> EvaluationHelper::process_inference_predictions() const
{
    std::vector<InferenceResult> prediction_results(dataset_.get_amount_of_classes());
    for (size_t i = 0; i < predictions_.size(); ++i)
    {
        auto const &prediction = predictions_[i];
        auto const &cur_data = dataset_.get_data_for_inference()[i];

        if (!prediction.votes_)
            ++prediction_results[cur_data.first].false_negatives_;
        else if (prediction.predicted_class_ != cur_data.first)
            ++prediction_results[cur_data.first].false_positives_;
        else  //votes have been cast and predicted class == correct class
            ++prediction_results[cur_data.first].true_positives_;
    }

    // Calculate true negatives.
    for (auto &res : prediction_results)
    {
        res.true_negatives_ = predictions_.size() - res.true_positives_ - res.false_negatives_ - res.false_positives_;
    }

    return prediction_results;
}


void InferenceResultsProcessor::process_inference_results(
    const std::vector<knp::core::messaging::SpikeMessage> &spikes,
    knp::framework::data_processing::classification::Dataset const &dataset)
{
    EvaluationHelper helper(dataset, inference_results_);
    knp::core::messaging::SpikeData firing_neuron_indices;
    auto spikes_iter = spikes.begin();

    for (size_t step = 0; step < dataset.get_steps_required_for_inference(); ++step)
    {
        while (spikes_iter != spikes.end() && spikes_iter->header_.send_time_ == step)
        {
            firing_neuron_indices.insert(
                firing_neuron_indices.end(), spikes_iter->neuron_indexes_.begin(), spikes_iter->neuron_indexes_.end());
            ++spikes_iter;
        }
        helper.process_spikes(firing_neuron_indices, step);
        firing_neuron_indices.clear();
    }

    inference_results_ = helper.process_inference_predictions();
}


void InferenceResultsProcessor::write_inference_results_to_stream_as_csv(std::ostream &results_stream)
{
    results_stream << "CLASS,TOTAL_VOTES,TRUE_POSITIVES,FALSE_NEGATIVES,FALSE_POSITIVES,TRUE_NEGATIVES,PRECISION,"
                      "RECALL,PREVALENCE,ACCURACY,F_MEASURE\n";
    for (size_t label = 0; label < inference_results_.size(); ++label)
    {
        auto const &prediction = inference_results_[label];
        const float precision = get_precision(prediction.true_positives_, prediction.false_positives_);
        const float recall = get_recall(prediction.true_positives_, prediction.false_positives_);
        const float prevalence = get_prevalence(
            prediction.true_positives_, prediction.false_negatives_, prediction.false_positives_,
            prediction.true_negatives_);
        const float accuracy = get_accuracy(
            prediction.true_positives_, prediction.false_negatives_, prediction.false_positives_,
            prediction.true_negatives_);
        const float f_score = get_f_score(precision, recall);

        results_stream << label << ',' << prediction.get_total_votes() << ',' << prediction.true_positives_ << ','
                       << prediction.false_negatives_ << ',' << prediction.false_positives_ << ','
                       << prediction.true_negatives_ << ',' << precision << ',' << recall << ',' << prevalence << ','
                       << accuracy << ',' << f_score << std::endl;
    }
}

}  // namespace knp::framework::inference_evaluation::classification
