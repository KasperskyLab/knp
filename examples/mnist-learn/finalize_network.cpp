/**
 * @file finalize_network.cpp
 * @brief Function for finalizing trained network.
 * @kaspersky_support D. Postnikov
 * @date 03.02.2026
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


#include "finalize_network.h"

#include "models/altai/finalize_network.h"
#include "models/blifat/finalize_network.h"


void finalize_network(AnnotatedNetwork& network, const ModelDescription& model_desc)
{
    switch (model_desc.type_)
    {
        case SupportedModelType::BLIFAT:
        {
            finalize_network_blifat(network, model_desc);
            break;
        }
        case SupportedModelType::AltAI:
        {
            finalize_network_altai(network, model_desc);
            break;
        }
        default:
            throw std::runtime_error("Unknown model type.");
    }
}
