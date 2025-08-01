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


namespace knp::framework::data_processing::classification
{

void Dataset::split(float split_percent)
{
    size_t split_beginning = static_cast<float>(data_for_training_.size()) * split_percent + 0.5F;
    for (size_t i = split_beginning; i < data_for_training_.size(); ++i)
        data_for_inference_.emplace_back(std::move(data_for_training_[i]));
    data_for_training_.erase(data_for_training_.begin() + split_beginning, data_for_training_.end());

    /*
     * The idea is that, if is too big for required training amount, then inference will be bigger than
     * training, so to compensate we make inference smaller, according to split.
     */
    if (required_training_amount_ < data_for_training_.size())
    {
        data_for_training_.resize(required_training_amount_);
        data_for_inference_.resize(
            static_cast<size_t>(static_cast<float>(data_for_training_.size()) / split_percent) -
            data_for_training_.size());
        steps_required_for_training_ = steps_per_frame_ * data_for_training_.size();
        steps_required_for_inference_ = steps_per_frame_ * data_for_inference_.size();
    }
    else
    {
        steps_required_for_training_ = steps_per_frame_ * required_training_amount_;
        steps_required_for_inference_ = steps_per_frame_ * data_for_inference_.size();
    }
}

}  // namespace knp::framework::data_processing::classification
