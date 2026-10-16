/**
 * @file message_header.cpp
 * @brief Implementation of message header I/O operators.
 * @kaspersky_support Vartenkov A.
 * @date 23.03.2023
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

#include <knp/core/messaging/message_header.h>
#include <knp/core/types.h>


namespace knp::core::messaging
{

std::istream &operator>>(std::istream &stream, MessageHeader &header)
{
    stream >> header.sender_uid_ >> header.send_time_;
    return stream;
}

std::ostream &operator<<(std::ostream &stream, const MessageHeader &header)
{
    stream << header.sender_uid_ << " " << header.send_time_;
    return stream;
}

}  // namespace knp::core::messaging
