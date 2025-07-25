/**
 * @file classification.cpp
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

#include <knp/core/messaging/messaging.h>
#include <knp/framework/inference_evaluation/classification.h>
#include <knp/framework/inference_evaluation/perfomance_metrics.h>

#include <algorithm>
#include <utility>


namespace knp::framework::inference_evaluation::classification
{

class EvaluationHelper
{
public:
    EvaluationHelper(
        const knp::framework::data_processing::classification::Dataset &dataset, size_t classes_amount,
        size_t steps_per_class)
        : classes_amount_(classes_amount),
          steps_per_class_(steps_per_class),
          prediction_votes_(classes_amount, 0),
          dataset_(dataset)
    {
    }

    void process_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices, size_t step);

    std::vector<InferenceResultForClass> process_inference_predictions() const;

private:
    void write_inference_results_to_stream(
        std::ostream &results_stream, const std::vector<InferenceResultForClass> &prediction_results) const;

    struct Prediction
    {
        size_t predicted_class_ = 0;
        size_t votes_ = 0;
    };
    std::vector<Prediction> predictions_;
    const size_t classes_amount_, steps_per_class_;
    std::vector<size_t> prediction_votes_;
    const knp::framework::data_processing::classification::Dataset &dataset_;
};


void EvaluationHelper::process_spikes(const knp::core::messaging::SpikeData &firing_neuron_indices, size_t step)
{
    for (auto i : firing_neuron_indices) ++prediction_votes_[i % classes_amount_];
    if (!((step + 1) % steps_per_class_))
    {
        size_t n_max = 0;
        size_t predicted_state = 0;

        for (size_t i = 0; i < classes_amount_; ++i)
        {
            if (prediction_votes_[i] > n_max)
            {
                n_max = prediction_votes_[i];
                predicted_state = i;
            }
        }
        predictions_.emplace_back(Prediction{predicted_state, n_max});
        std::fill(prediction_votes_.begin(), prediction_votes_.end(), 0);
    }
}


std::vector<InferenceResultForClass> EvaluationHelper::process_inference_predictions() const
{
    std::vector<InferenceResultForClass> prediction_results(classes_amount_);
    for (size_t i = 0; i < predictions_.size(); ++i)
    {
        auto const &prediction = predictions_[i];
        auto const &cur_data = dataset_.data_for_inference_[i];
        ++prediction_results[cur_data.first].total_;
        if (!prediction.votes_)
            ++prediction_results[cur_data.first].no_votes_;
        else if (prediction.predicted_class_ != cur_data.first)
            ++prediction_results[cur_data.first].incorrectly_predicted_;
        else  //votes has been cast and predicted class == correct class
            ++prediction_results[cur_data.first].correctly_predicted_;
    }
    return prediction_results;
}


std::vector<InferenceResultForClass> process_inference_results(
    const std::vector<knp::core::messaging::SpikeMessage> &spikes,
    knp::framework::data_processing::classification::Dataset const &dataset, size_t classes_amount,
    size_t steps_per_class)
{
    EvaluationHelper helper(dataset, classes_amount, steps_per_class);
    knp::core::messaging::SpikeData firing_neuron_indices;
    auto spikes_iter = spikes.begin();

    for (size_t step = 0; step < dataset.steps_required_for_inference_; ++step)
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

    return helper.process_inference_predictions();
}

void write_inference_results_to_stream_as_csv(
    std::ostream &results_stream, const std::vector<InferenceResultForClass> &inference_results)
{
    results_stream << "CLASS,TOTAL_VOTES,CORRECT_VOTES,INCORRECT_VOTES,NO_VOTES,PRECISION,RECALL,F_MEASURE\n";
    for (size_t label = 0; label < inference_results.size(); ++label)
    {
        auto const &prediction = inference_results[label];
        float precision = get_precision(prediction.correctly_predicted_, prediction.incorrectly_predicted_);
        float recall = get_recall(prediction.correctly_predicted_, prediction.incorrectly_predicted_);
        float f_measure = get_f_measure(precision, recall);
        results_stream << label << ',' << prediction.total_ << ',' << prediction.correctly_predicted_ << ','
                       << prediction.incorrectly_predicted_ << ',' << prediction.no_votes_ << ',' << precision << ','
                       << recall << ',' << f_measure << std::endl;
    }
}

}  // namespace knp::framework::inference_evaluation::classification
