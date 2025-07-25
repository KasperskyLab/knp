/**
 * @file data_processing_test.cpp
 * @brief Data processing test
 * @kaspersky_support D. Postnikov
 * @date 21.08.2025
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

#include <tests_common.h>


TEST(DataProcessing, ImageClassification)
{
    constexpr size_t training_amount = 10, image_size = 1, steps_per_image = 1;
    constexpr float dataset_split = 2.F / 3.F;
    std::stringstream images_stream("\x01\x02\x03");
    std::stringstream labels_stream("0\n1\n2\n");
    auto dataset = knp::framework::data_processing::classification::images::process_data(
        images_stream, labels_stream, training_amount, dataset_split, image_size, steps_per_image,
        [](std::vector<uint8_t> const&) -> std::vector<bool> { return {true}; });

    ASSERT_EQ(dataset.image_size_, image_size);
    ASSERT_EQ(dataset.steps_per_class_, steps_per_image);
    ASSERT_EQ(dataset.steps_required_for_training_, 10);
    ASSERT_EQ(dataset.steps_required_for_inference_, 1);

    ASSERT_EQ(dataset.data_for_training_.size(), 2);
    ASSERT_EQ(dataset.data_for_training_[0].first, 0);
    ASSERT_EQ(dataset.data_for_training_[0].second.size(), 1);
    ASSERT_EQ(dataset.data_for_training_[0].second[0], true);
    ASSERT_EQ(dataset.data_for_training_[1].first, 1);
    ASSERT_EQ(dataset.data_for_training_[1].second.size(), 1);
    ASSERT_EQ(dataset.data_for_training_[1].second[0], true);

    ASSERT_EQ(dataset.data_for_inference_.size(), 1);
    ASSERT_EQ(dataset.data_for_inference_[0].first, 2);
    ASSERT_EQ(dataset.data_for_inference_[0].second.size(), 1);
    ASSERT_EQ(dataset.data_for_inference_[0].second[0], true);
}
