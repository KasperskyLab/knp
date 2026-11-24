/**
 * @file uid.cpp
 * @brief UID implementation.
 * @kaspersky_support Artiom N.
 * @date 22.05.20
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

#include <knp/core/uid.h>

#include <atomic>


namespace knp::core
{

namespace
{
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
volatile ::std::atomic_uint64_t tag_counter = 1;
}  // namespace


::boost::uuids::uuid continuously_uid_generator::operator()() const
{
    ::boost::uuids::uuid result = boost::uuids::nil_uuid();
    uint64_t c_value = tag_counter++;
    constexpr auto value_size = sizeof(c_value);

    static_assert(
        sizeof(decltype(tag_counter)::value_type) <= ::boost::uuids::uuid::static_size(), "UID counter size error.");
    for (size_t i = 0; i < value_size; ++i)
    {
        auto digit = c_value & 0xffU;
        c_value >>= 8;
        *(result.begin() + i) = static_cast<uint8_t>(digit);
    }

    return result;
}


void continuously_uid_generator::reset(uint64_t initial_value)
{
    tag_counter = initial_value;
}

}  // namespace knp::core
