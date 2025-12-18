/**
 * @file dataset.h
 * @brief Header file for classification dataset definition.
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


/**
 * @brief Namespace for data processing.
 */
namespace knp::framework::data_processing
{

/**
 * @brief Namespace for classification data processing.
 */
namespace classification
{

/**
 * @brief The `Dataset` class is the base class for datasets.
 * @details A dataset is supposed to be abstracted from its actual processing and characteristics, such as size.
 * The size of the dataset is not a crucial factor, as it is handled during the splitting process. The correct
 * workflow would be to first process a dataset, then split it, and finally use it for your purposes.
 * Splitting the dataset is important because it calculates the number of steps required for inference and/or training.
 */
class KNP_DECLSPEC Dataset
{
protected:
    /**
     * @brief Destructor.
     */
    virtual ~Dataset() = default;

    /**
     * @brief Construct a default `Dataset` object.
     */
    Dataset() = default;

    /**
     * @brief Construct a new `Dataset` object by copying the contents of another `Dataset` object.
     * @param dataset the `Dataset` object to copy.
     */
    Dataset(const Dataset& dataset) = default;

    /**
     * @brief Assign the contents of another `Dataset` object to new object.
     * @param dataset the `Dataset` object to copy.
     * @return reference to new `Dataset` object.
     */
    Dataset& operator=(const Dataset& dataset) = default;

    /**
     * @brief Construct a new `Dataset` object by moving the contents of another `Dataset` object.
     * @param dataset the `Dataset` object to move.
     */
    Dataset(Dataset&& dataset) = default;

    /**
     * @brief Assign the contents of another `Dataset` object to new object, transferring ownership.
     * @param dataset the `Dataset` object to move.
     * @return reference to new `Dataset` object.
     */
    Dataset& operator=(Dataset&& dataset) = default;

public:
    /**
     * @brief Split dataset into training and inference based on given requirements.
     * @pre Sum of given parameters should be less or equal to size of whole dataset.
     * @param frames_for_training Amount of frames for training.
     * @param frames_for_inference Amount of frames for inference.
     */
    void split(size_t frames_for_training, size_t frames_for_inference);


    /**
     * @brief Get training data, consisting of pairs of labels and frames.
     * @return constant reference to the training data.
     * @note The returned data is the result of the @ref split function, which allocates data for training.
     */
    [[nodiscard]] inline auto const& get_data_for_training() const { return data_for_training_; }

    /**
     * @brief Get inference data, consisting of pairs of labels and frames.
     * @return constant reference to the inference data.
     * @note The returned data is the result of the @ref split function, which allocates data for inference.
     */
    [[nodiscard]] inline auto const& get_data_for_inference() const { return data_for_inference_; }

    /**
     * @brief Get the number of steps each frame is distributed to.
     * @return number of steps per frame.
     */
    [[nodiscard]] inline size_t get_steps_per_frame() const { return steps_per_frame_; }

    /**
     * @brief Get the total number of steps required for training.
     * @return number of steps required for training, calculated based on the training data and steps per frame.
     */
    [[nodiscard]] inline size_t get_steps_required_for_training() const { return steps_required_for_training_; }

    /**
     * @brief Get the total number of steps required for inference.
     * @return number of steps required for inference, calculated based on the inference data and steps per frame.
     */
    [[nodiscard]] inline size_t get_steps_required_for_inference() const { return steps_required_for_inference_; }

    /**
     * @brief Get the user-specified amount of training data required.
     * @return required training amount, which may affect the allocation of data for inference.
     */
    [[nodiscard]] inline size_t get_required_training_amount() const { return required_training_amount_; }

    /**
     * @brief Get the number of classes in the dataset.
     * @return number of classes.
     */
    [[nodiscard]] inline size_t get_amount_of_classes() const { return classes_amount_; }

    /**
     * @brief The structure represents a class instance in the form of spikes, distributed over multiple steps.
     * @details This structure encapsulates the spike data for a class instance, which is transmitted over a series of
     * steps. For example, an image might be sent over 20 steps, with each step representing a subset of the image data.
     * The structure stores a vector of boolean values, where each value indicates whether a spike should be sent at a
     * particular step. The length of this vector is determined by the product of the steps per frame and the size of
     * the class instance data.
     */
    struct Frame
    {
        // cppcheck-suppress unusedStructMember
        /**
         * @brief A vector of boolean values representing the spike pattern for this frame.
         * @note The length of this vector is equal to the number of steps per frame multiplied by the size of the class
         * instance data.
         */
        std::vector<bool> spikes_;
    };

protected:
    /**
     * @brief Training data, consisting of pairs of labels and frames.
     * @note This vector is modified by the @ref split function to allocate data for training.
     */
    std::vector<std::pair<unsigned, Frame>> data_for_training_;

    /**
     * @brief Inference data, consisting of pairs of labels and frames.
     * @note This vector is modified by the @ref split function to allocate data for inference.
     */
    std::vector<std::pair<unsigned, Frame>> data_for_inference_;

    /**
     * @brief Number of steps each frame is distributed to.
     */
    size_t steps_per_frame_ = 0;

    /**
     * @brief Total number of steps required for training, calculated based on @ref data_for_training_ and @ref
     * steps_per_frame_.
     */
    size_t steps_required_for_training_ = 0;

    /**
     * @brief Total number of steps required for inference, calculated based on @ref data_for_inference_ and @ref
     * steps_per_frame_.
     */
    size_t steps_required_for_inference_ = 0;

    /**
     * @brief User-specified amount of training data required.
     * @note If this value is less than the actual size of @ref data_for_training_, the @ref split function adjusts the
     * inference data accordingly.
     */
    size_t required_training_amount_ = 0;

    /**
     * @brief Number of classes in the dataset.
     */
    size_t classes_amount_ = 0;
};

}  // namespace classification

}  //namespace knp::framework::data_processing
