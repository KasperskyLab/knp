/**
 * @file base.h
 * @brief Network validator base class.
 * @kaspersky_support David P.
 * @date 02.04.2026
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

#include <knp/framework/network.h>

#include <string>


/**
 * @brief Network validators namespace.
 */
namespace knp::framework::network_validators
{

/**
 * @brief Base class for all validators. Every validator should publicly inherit this class.
 */
class KNP_DECLSPEC Base
{
public:
    /**
     * @brief Empty constructor.
     */
    Base() = default;


    /**
     * @brief Copy constructor.
     */
    Base(const Base&) = default;


    /**
     * @brief Copy asignment operator.
     * @return Original class instance.
     */
    Base& operator=(const Base&) = default;

    /**
     * @brief Move constructor.
     */
    Base(Base&&) = default;

    /**
     * @brief Move asignment operator.
     * @return Original class instance.
     */
    Base& operator=(Base&&) = default;


    /**
     * @brief Virtual destructor.
     */
    virtual ~Base() = default;


    /**
     * @brief Get name of validator for logs.
     * @return Validator name.
     */
    [[nodiscard]] virtual std::string get_name() const = 0;


    /**
     * @brief Run validation on specified network.
     * @note If there is any kind of warning or error, result should be 'false'.
     * @param network Network for validation.
     * @return Result of validation.
     */
    [[nodiscard]] virtual bool run_validation(const Network& network) = 0;
};
}  //namespace knp::framework::network_validators
