/**
 * @file model_desc.cpp
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

#include "model_desc.h"

std::ostream& operator<<(std::ostream& stream, ModelDescription const& desc)
{
    stream << "Model type: ";
    switch (desc.type_)
    {
        case SupportedModelType::BLIFAT:
            stream << "BLIFAT";
            break;
        case SupportedModelType::AltAI:
            stream << "AltAI";
            break;
        default:
            throw std::runtime_error("Unknown model type.");
    }
    stream << "\n";

    stream << "Train images amount: " << desc.train_images_amount_ << "\n";
    stream << "Inferece images amount: " << desc.inference_images_amount_ << "\n";

    stream << "Images file path: " << desc.images_file_path_ << "\n";
    stream << "Labels file path: " << desc.labels_file_path_ << "\n";

    stream << "Backend path: " << desc.backend_path_ << "\n";

    if (desc.log_path_.empty())
        stream << "Logs wont be saved.\n";
    else
        stream << "Log path: " << desc.log_path_ << "\n";

    if (desc.model_saving_path_.empty())
        stream << "Model wont be saved.\n";
    else
        stream << "Model saving path: " << desc.model_saving_path_ << "\n";

    return stream;
}
