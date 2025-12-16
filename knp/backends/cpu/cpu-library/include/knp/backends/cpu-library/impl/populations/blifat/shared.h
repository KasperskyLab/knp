/**
 * @file shared.h
 * @kaspersky_support Postnikov D.
 * @date 12.12.2025
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
#pragma once

#include <knp/core/population.h>

namespace knp::backends::cpu::populations::impl::blifat
{

/**
 * @brief BLIFAT neuron shortcut.
 */
using BlifatNeuron = knp::neuron_traits::BLIFATNeuron;

/**
 * @brief STDP BLIFAT neuron shortcut.
 */
using STDPBlifatNeuron = knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron;

}  //namespace knp::backends::cpu::populations::impl::blifat
