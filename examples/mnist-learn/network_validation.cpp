/**
 * @file network_validation.cpp
 * @brief Function to validate network.
 * @kaspersky_support D. Postnikov
 * @date 03.04.2026
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

#include "network_validation.h"

#include <knp/framework/network_validation/runner.h>
#include <knp/framework/network_validation/validators/connectivity.h>


void validate_network(const knp::framework::Network& network)
{
    knp::framework::network_validation::Runner validations_runner;
    validations_runner.add_validator("Connectivity validator", knp::framework::network_validation::Connectivity());
    validations_runner.run_validators(network);
}
