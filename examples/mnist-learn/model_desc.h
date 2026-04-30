/**
 * @file model_desc.h
 * @brief Model description.
 * @kaspersky_support D. Postnikov
 * @date 03.02.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include <filesystem>


/**
 * @brief Supported model types for neural networks.
 * 
 * @details Enumerates the neuron models available for MNIST digit recognition.
 */
enum class SupportedModelType
{
    /**
     * @brief BLIFAT neuron model implementation.
     */
    BLIFAT,
    /**
     * @brief AltAI neuron model implementation.
     */
    AltAI
};


/**
 * @brief Configuration structure for neural network models.
 * 
 * @details Contains all parameters that can be configured for the MNIST learning example,
 * including model type, dataset sizes, file paths, and logging settings.
 * 
 * @note Parameters can be changed via command line.
 */
struct ModelDescription
{
    /**
     * @brief Selected model type.
     */
    // cppcheck-suppress unusedStructMember
    SupportedModelType type_;

    /**
     * @brief Number of images to use for training.
     */
    // cppcheck-suppress unusedStructMember
    size_t train_images_amount_;

    /**
     * @brief Number of images to use for inference.
     */
    // cppcheck-suppress unusedStructMember
    size_t inference_images_amount_;

    /**
     * @brief Path to the binary images file.
     */
    std::filesystem::path images_file_path_;

    /**
     * @brief Path to the labels file corresponding to the images.
     */
    std::filesystem::path labels_file_path_;

    /**
     * @brief Path to backend library for training operations.
     * 
     * @note Platform specific name parts (for example, .so or .dll) should be absent from the path.
     */
    std::filesystem::path training_backend_path_;

    /**
     * @brief Path to backend library for inference operations.
     * 
     * @note Platform specific name parts (for example, .so or .dll) should be absent from the path.
     */
    std::filesystem::path inference_backend_path_;

    /**
     * @brief Path to directory where detailed logs will be saved.
     */
    std::filesystem::path log_path_;

    /**
     * @brief Path to directory where trained model will be saved in SONATA format.
     */
    std::filesystem::path model_saving_path_;
};


/**
 * @brief Stream output operator for @ref ModelDescription structure.
 * 
 * @details Provides a human-readable string representation of the @ref ModelDescription object, displaying all configuration 
 * parameters including model type, dataset settings, file paths, and logging configuration.
 * 
 * @param stream output stream to write the formatted description to.
 * @param desc model description containing configuration parameters.
 * @return reference to the output stream.
 */
std::ostream& operator<<(std::ostream& stream, ModelDescription const& desc);
