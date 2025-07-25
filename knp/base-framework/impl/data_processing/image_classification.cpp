/**
 * @file image_classification.cpp
 * @brief Processing of dataset
 * @kaspersky_support D. Postnikov
 * @date 14.07.2025
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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

#include <knp/framework/data_processing/image_classification.h>

namespace knp::framework::data_processing::classification::images
{

std::vector<bool> incrementing_image_to_spikes(
    const std::vector<uint8_t> &image, std::vector<float> &states, size_t steps_per_image, size_t active_steps,
    size_t image_size, float state_increment_factor)
{
    std::vector<bool> ret;
    ret.reserve(steps_per_image * image_size);

    for (size_t i = 0; i < active_steps; ++i)
    {
        ret.insert(ret.end(), image_size, false);
        for (size_t l = 0; l < image_size; ++l)
        {
            states[l] += state_increment_factor * static_cast<float>(image[l]);
            if (states[l] >= 1.F)
            {
                ret[ret.size() - image_size + l] = true;
                --states[l];
            }
        }
    }

    ret.insert(ret.end(), (steps_per_image - active_steps) * image_size, false);

    return ret;
}


std::function<std::vector<bool>(std::vector<uint8_t> const &)> make_incrementing_image_to_spikes_converter(
    size_t steps_per_image, size_t active_steps, size_t image_size, float state_increment_factor,
    std::vector<float> &&states)
{
    return [=, &states](std::vector<uint8_t> const &image) -> std::vector<bool>
    {
        return incrementing_image_to_spikes(
            image, states, steps_per_image, active_steps, image_size, state_increment_factor);
    };
}

void split_dataset(Dataset &dataset, float dataset_split)
{
    size_t split_beginning =
        static_cast<size_t>(static_cast<float>(dataset.data_for_training_.size()) * dataset_split + 0.5F);
    for (size_t i = split_beginning; i < dataset.data_for_training_.size(); ++i)
        dataset.data_for_inference_.emplace_back(std::move(dataset.data_for_training_[i]));
    dataset.data_for_training_.erase(
        dataset.data_for_training_.begin() + split_beginning, dataset.data_for_training_.end());
}

Dataset process_data(
    std::istream &images_stream, std::istream &labels_stream, size_t training_amount, float dataset_split,
    size_t image_size, size_t steps_per_image,
    std::function<std::vector<bool>(std::vector<uint8_t> const &)> const &image_to_spikes)
{
    Dataset dataset;

    dataset.image_size_ = image_size;
    dataset.steps_per_class_ = steps_per_image;

    {  // Process dataset.
        std::vector<uint8_t> image_reading_buffer(image_size, 0);

        while (images_stream.good() && labels_stream.good())
        {
            images_stream.read(reinterpret_cast<char *>(image_reading_buffer.data()), image_size);
            auto spikes_frame = image_to_spikes(image_reading_buffer);

            std::string str;
            if (!std::getline(labels_stream, str).good()) break;
            int label = std::stoi(str);

            // Push to training data set because we dont know dataset size yet for a split
            dataset.data_for_training_.push_back({label, std::move(spikes_frame)});
        }

        split_dataset(dataset, dataset_split);
    }

    /*
     * The idea is that, if dataset is too big for required training amount, then inference will be bigger than
     * training, so to compensate we make inference smaller, according to dataset_split.
     */
    if (training_amount < dataset.data_for_training_.size())
    {
        dataset.data_for_training_.resize(training_amount);
        dataset.data_for_inference_.resize(
            static_cast<size_t>(static_cast<float>(dataset.data_for_training_.size()) / dataset_split) -
            dataset.data_for_training_.size());
        dataset.steps_required_for_training_ = steps_per_image * dataset.data_for_training_.size();
        dataset.steps_required_for_inference_ = steps_per_image * dataset.data_for_inference_.size();
    }
    else
    {
        dataset.steps_required_for_training_ = steps_per_image * training_amount;
        dataset.steps_required_for_inference_ = steps_per_image * dataset.data_for_inference_.size();
    }

    return dataset;
}


std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_training_labels_generator(const Dataset &dataset)
{
    return [&dataset](knp::core::Step step)
    {
        knp::core::messaging::SpikeData message;
        message.push_back(
            dataset.data_for_training_[(step / dataset.steps_per_class_) % dataset.data_for_training_.size()].first);
        return message;
    };
}


std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_training_images_spikes_generator(
    const Dataset &dataset)
{
    return [&dataset](knp::core::Step step)
    {
        knp::core::messaging::SpikeData message;
        auto const &data =
            dataset.data_for_training_[(step / dataset.steps_per_class_) % dataset.data_for_training_.size()].second;
        size_t local_step = step % dataset.steps_per_class_;
        size_t frame_start = local_step * dataset.image_size_;
        for (size_t i = frame_start; i < frame_start + dataset.image_size_; ++i)
        {
            if (data[i]) message.push_back(i - frame_start);
        }
        return message;
    };
}


std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_inference_images_spikes_generator(
    Dataset const &dataset)
{
    return [&dataset](knp::core::Step step)
    {
        knp::core::messaging::SpikeData message;
        auto const &data =
            dataset.data_for_inference_[(step / dataset.steps_per_class_) % dataset.data_for_inference_.size()].second;
        size_t local_step = step % dataset.steps_per_class_;
        size_t frame_start = local_step * dataset.image_size_;
        for (size_t i = frame_start; i < frame_start + dataset.image_size_; ++i)
        {
            if (data[i]) message.push_back(i - frame_start);
        }
        return message;
    };
}

}  //namespace knp::framework::data_processing::classification::images
