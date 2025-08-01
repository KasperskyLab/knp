/**
 * @file image.cpp
 * @brief Processing of dataset of images.
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

#include <knp/framework/data_processing/classification/image.h>


namespace knp::framework::data_processing::classification::images
{

void Dataset::process_labels_and_images(
    std::istream &images_stream, std::istream &labels_stream, size_t training_amount, size_t classes_amount,
    size_t image_size, size_t steps_per_image,
    std::function<Frame(std::vector<uint8_t> const &)> const &image_to_spikes)
{
    image_size_ = image_size;
    steps_per_frame_ = steps_per_image;
    required_training_amount_ = training_amount;
    classes_amount_ = classes_amount;

    std::vector<uint8_t> image_reading_buffer(image_size, 0);

    while (images_stream.good() && labels_stream.good())
    {
        images_stream.read(reinterpret_cast<char *>(image_reading_buffer.data()), image_size);
        auto spikes_frame = image_to_spikes(image_reading_buffer);

        std::string str;
        if (!std::getline(labels_stream, str).good()) break;
        int label = std::stoi(str);

        // Push to training data set because we dont know dataset size yet for a split
        data_for_training_.push_back({label, std::move(spikes_frame)});
    }
}


std::function<knp::core::messaging::SpikeData(knp::core::Step)> Dataset::make_training_labels_generator() const
{
    return [this](knp::core::Step step)
    {
        knp::core::messaging::SpikeData message;

        size_t frame_index = step / steps_per_frame_;
        size_t looped_frame_index = frame_index % data_for_training_.size();

        message.push_back(data_for_training_[looped_frame_index].first);
        return message;
    };
}


std::function<knp::core::messaging::SpikeData(knp::core::Step)> Dataset::make_training_images_spikes_generator() const
{
    return [this](knp::core::Step step)
    {
        knp::core::messaging::SpikeData message;

        size_t frame_index = step / steps_per_frame_;
        size_t looped_frame_index = frame_index % data_for_training_.size();

        auto const &data = data_for_training_[looped_frame_index].second.spikes_;

        size_t local_step = step % steps_per_frame_;
        size_t frame_start = local_step * image_size_;

        for (size_t i = frame_start; i < frame_start + image_size_; ++i)
        {
            if (data[i]) message.push_back(i - frame_start);
        }
        return message;
    };
}


std::function<knp::core::messaging::SpikeData(knp::core::Step)> Dataset::make_inference_images_spikes_generator() const
{
    return [this](knp::core::Step step)
    {
        knp::core::messaging::SpikeData message;

        size_t frame_index = step / steps_per_frame_;
        size_t looped_frame_index = frame_index % data_for_inference_.size();

        auto const &data = data_for_inference_[looped_frame_index].second.spikes_;

        size_t local_step = step % steps_per_frame_;
        size_t frame_start = local_step * image_size_;

        for (size_t i = frame_start; i < frame_start + image_size_; ++i)
        {
            if (data[i]) message.push_back(i - frame_start);
        }
        return message;
    };
}


std::function<Dataset::Frame(std::vector<uint8_t> const &)> Dataset::make_incrementing_image_to_spikes_converter(
    size_t active_steps, float state_increment_factor) const

{
    std::vector<float> states;
    return [this, active_steps, state_increment_factor, states](std::vector<uint8_t> const &image) mutable -> Frame
    {
        if (!states.size()) states.resize(image_size_, 0.F);

        std::vector<bool> ret;
        ret.reserve(steps_per_frame_ * image_size_);

        for (size_t i = 0; i < active_steps; ++i)
        {
            ret.insert(ret.end(), image_size_, false);
            for (size_t l = 0; l < image_size_; ++l)
            {
                states[l] += state_increment_factor * static_cast<float>(image[l]);
                if (states[l] >= 1.F)
                {
                    ret[ret.size() - image_size_ + l] = true;
                    --states[l];
                }
            }
        }

        ret.insert(ret.end(), (steps_per_frame_ - active_steps) * image_size_, false);

        return {std::move(ret)};
    };
}

}  // namespace knp::framework::data_processing::classification::images
