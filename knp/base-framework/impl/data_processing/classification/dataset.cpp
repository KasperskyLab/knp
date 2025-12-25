/**
 * @file dataset.cpp
 * @brief Definition of classification dataset.
 * @kaspersky_support D. Postnikov
 * @date 29.07.2025
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

#include <knp/framework/data_processing/classification/dataset.h>

#include <spdlog/spdlog.h>


namespace knp::framework::data_processing::classification
{

void Dataset::split(size_t frames_for_training, size_t frames_for_inference)
{
    if (dataset_.size() < frames_for_inference + frames_for_training)
    {
        SPDLOG_ERROR(
            "Incorrect split size. Dataset is too small. Required {} frames for training, and {} frames for inference, "
            "while dataset only have {} frames.",
            frames_for_training, frames_for_inference, dataset_.size());
        throw std::runtime_error("Dataset too small.");
    }

    frames_amount_for_training_ = frames_for_training;
    frames_amount_for_inference_ = frames_for_inference;
}

auto Dataset::get_data_for_training() const -> std::span<const NamedFrame>
{
    return {dataset_.begin(), dataset_.begin() + frames_amount_for_training_};
}

auto Dataset::get_data_for_inference() const -> std::span<const NamedFrame>
{
    return {
        dataset_.begin() + frames_amount_for_training_,
        dataset_.begin() + frames_amount_for_training_ + frames_amount_for_inference_};
}

}  // namespace knp::framework::data_processing::classification
