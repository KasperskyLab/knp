/**
 * @file data_processing_test.cpp
 * @brief Data processing test.
 * @kaspersky_support D. Postnikov
 * @date 21.08.2025
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

#include <knp/framework/data_processing/classification/image.h>

#include <tests_common.h>


TEST(DataProcessing, ImageClassification)
{
    constexpr size_t training_amount = 2, inference_amount = 1, classes_amount = 3, image_size = 1, steps_per_image = 1;
    std::stringstream images_stream("\x01\x02\x03");
    std::stringstream labels_stream("0\n1\n2\n");
    knp::framework::data_processing::classification::images::Dataset dataset;
    dataset.process_labels_and_images(
        images_stream, labels_stream, training_amount + inference_amount, classes_amount, image_size, steps_per_image,
        [](std::vector<uint8_t> const&) -> knp::framework::data_processing::classification::Dataset::Frame
        { return {{true}}; });
    dataset.split(training_amount, inference_amount);

    ASSERT_EQ(dataset.get_image_size(), image_size);
    ASSERT_EQ(dataset.get_amount_of_classes(), classes_amount);
    ASSERT_EQ(dataset.get_steps_per_frame(), steps_per_image);
    ASSERT_EQ(dataset.get_steps_amount_for_training(), training_amount);
    ASSERT_EQ(dataset.get_steps_amount_for_inference(), inference_amount);

    auto [training_begin, training_end] = dataset.get_data_for_training();
    ASSERT_EQ(std::distance(training_begin, training_end), training_amount);
    ASSERT_EQ(training_begin[0].first, 0);
    ASSERT_EQ(training_begin[0].second.spikes_.size(), 1);
    ASSERT_EQ(training_begin[0].second.spikes_[0], true);
    ASSERT_EQ(training_begin[1].first, 1);
    ASSERT_EQ(training_begin[1].second.spikes_.size(), 1);
    ASSERT_EQ(training_begin[1].second.spikes_[0], true);

    auto [inference_begin, inference_end] = dataset.get_data_for_inference();
    ASSERT_EQ(std::distance(inference_begin, inference_end), inference_amount);
    ASSERT_EQ(inference_begin[0].first, 2);
    ASSERT_EQ(inference_begin[0].second.spikes_.size(), 1);
    ASSERT_EQ(inference_begin[0].second.spikes_[0], true);

    auto train_images_spikes_gen = dataset.make_training_images_spikes_generator();
    for (size_t i = 0; i < dataset.get_steps_amount_for_training(); ++i)
    {
        const auto res = train_images_spikes_gen(i);
        ASSERT_EQ(res.size(), 1);
        ASSERT_EQ(res[0], 0);
    }

    auto train_labels_gen = dataset.make_training_labels_generator();
    for (size_t i = 0; i < dataset.get_steps_amount_for_training(); ++i)
    {
        const auto res = train_labels_gen(i);
        ASSERT_EQ(res.size(), 1);
        ASSERT_EQ(res[0], i % std::distance(training_begin, training_end));
    }

    auto inf_images_spikes_gen = dataset.make_inference_images_spikes_generator();
    for (size_t i = 0; i < dataset.get_steps_amount_for_inference(); ++i)
    {
        const auto res = train_images_spikes_gen(i);
        ASSERT_EQ(res.size(), 1);
        ASSERT_EQ(res[0], 0);
    }
}
