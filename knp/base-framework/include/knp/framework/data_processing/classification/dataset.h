/**
 * @file dataset.h
 * @brief Definition of classification dataset.
 * @kaspersky_support D. Postnikov
 * @date 21.07.2025
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

#pragma once

#include <knp/core/core.h>
#include <knp/core/impexp.h>
#include <knp/core/messaging/messaging.h>

#include <utility>
#include <vector>


namespace knp::framework::data_processing::classification
{

/**
 * @brief A class that represents processed dataset.
 */
class KNP_DECLSPEC Dataset
{
protected:
    /**
     * @brief Destructor.
     */
    virtual ~Dataset() = default;

    /**
     * @brief Default constructor.
     */
    Dataset() = default;

    /**
     * @brief Copy constructor.
     * @param dataset Dataset.
     */
    Dataset(const Dataset& dataset) = default;

    /**
     * @brief Copy assignment operator.
     * @param dataset Dataset.
     * @return Dataset.
     */
    Dataset& operator=(const Dataset& dataset) = default;

    /**
     * @brief Move constructor.
     * @param dataset Dataset.
     */
    Dataset(Dataset&& dataset) = default;

    /**
     * @brief Move assignment operator.
     * @param dataset Dataset.
     * @return Dataset.
     */
    Dataset& operator=(Dataset&& dataset) = default;

public:
    /**
     * @brief Split dataset on train/inference.
     * @param split_percent Percentage that shows how to split dataset. Must be from 0 to 1, for example 0.8 will split
     * dataset so 80% dedicated for tranining and 20% for inference.
     */
    virtual void split(float split_percent);

    /**
     * @brief Get data for training.
     * @return Data for training.
     */
    [[nodiscard]] auto const& get_data_for_training() const { return data_for_training_; }

    /**
     * @brief Get data for inference.
     * @return Data for inference.
     */
    [[nodiscard]] auto const& get_data_for_inference() const { return data_for_inference_; }

    /**
     * @brief Get steps amount per frame.
     * @return Steps amount per frame.
     */
    [[nodiscard]] size_t get_steps_per_frame() const { return steps_per_frame_; }

    /**
     * @brief Get amount of steps required for training.
     * @return Amount of steps required for training.
     */
    [[nodiscard]] size_t get_steps_required_for_training() const { return steps_required_for_training_; }

    /**
     * @brief Get steps amount required for inference.
     * @return Steps amount required for inference.
     */
    [[nodiscard]] size_t get_steps_required_for_inference() const { return steps_required_for_inference_; }

    /**
     * @brief Get required training amount.
     * @return Required training amount.
     */
    [[nodiscard]] size_t get_required_training_amount() const { return required_training_amount_; }

    /**
     * @brief Get amount of classes..
     * @return Amount of classes.
     */
    [[nodiscard]] size_t get_amount_of_classes() const { return classes_amount_; }

    /**
     * @brief A struct that represents a class instance in form of spikes.
     * @details In classification we want to send class instance, that is converted to spikes form, in several steps.
     * For example and image can be sent over 20 steps. This struct represents class instance data on those several
     * steps. So it stores a vector of bools that represents where to send spikes over specified amount of steps. So
     * with image example, length of this vector would be equal to steps_per_frame * image_size.
     */
    struct Frame
    {
        std::vector<bool> spikes_;
    };

protected:
    /**
     * @brief Vector of pairs of label and frame.
     */
    std::vector<std::pair<unsigned, Frame>> data_for_training_;

    /**
     * @brief Vector of pairs of label and frame.
     */
    std::vector<std::pair<unsigned, Frame>> data_for_inference_;

    /**
     * @brief Amount of steps frame is discributed to.
     */
    size_t steps_per_frame_ = 0;

    /**
     * @brief Amount of steps required for training.
     */
    size_t steps_required_for_training_ = 0;

    /**
     * @brief Amount of steps required for inference.
     */
    size_t steps_required_for_inference_ = 0;

    /**
     * @brief Training amount required by user.
     */
    size_t required_training_amount_ = 0;

    /**
     * @brief Amount of classes.
     */
    size_t classes_amount_ = 0;
};

}  // namespace knp::framework::data_processing::classification
