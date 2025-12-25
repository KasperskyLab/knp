/**
 * @file message_endpoint_zmq_impl.cpp
 * @brief Message endpoint ZeroMQ implementation.
 * @kaspersky_support Artiom N.
 * @date 31.03.2023
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

#include "message_endpoint_zmq_impl.h"

#include <knp/meta/macro.h>

#include <spdlog/spdlog.h>

#include <memory>
#include <utility>

#include <zmq.hpp>


namespace knp::core::messaging::impl
{

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4455)
#endif
using std::chrono_literals::operator""ms;
#if defined(_MSC_VER)
#    pragma warning(pop)
#endif


MessageEndpointZMQImpl::MessageEndpointZMQImpl(zmq::socket_t &&sub_socket, zmq::socket_t &&pub_socket)
    : sub_socket_(std::move(sub_socket)), pub_socket_(std::move(pub_socket))
{
    SPDLOG_DEBUG("ZMQ message endpoint creating...");
}


void MessageEndpointZMQImpl::send_zmq_message(const std::vector<uint8_t> &data)
{
    send_zmq_message(data.data(), data.size());
}


void MessageEndpointZMQImpl::send_zmq_message(const void *data, size_t size)
{
    // `send_result` is `std::optional` and if it doesn't contain a value, EAGAIN is returned by the call.
    zmq::send_result_t result;
    try
    {
        SPDLOG_DEBUG("Endpoint sending message...");
        KNP_UNROLL_LOOP()
        do
        {
            SPDLOG_TRACE("Sending {} bytes...", size);
            result = pub_socket_.send(zmq::message_t(data, size), zmq::send_flags::dontwait);
            SPDLOG_TRACE("{} bytes were sent.", size);
        } while (!result.has_value());
    }
    catch (const zmq::error_t &e)
    {
        SPDLOG_CRITICAL(e.what());
        throw;
    }
}


std::optional<zmq::message_t> MessageEndpointZMQImpl::receive_zmq_message()
{
    zmq::message_t msg;
    // `recv_result` is `std::optional` and if it doesn't contain a value, `EAGAIN` is returned by the call.
    zmq::recv_result_t result;

    try
    {
        std::vector<zmq_pollitem_t> items = {
            zmq_pollitem_t{sub_socket_.handle(), 0, ZMQ_POLLIN, 0},
        };

        SPDLOG_DEBUG("Running poll() to receive a message...");
        // cppcheck-suppress "cppcheckError"
        if (zmq::poll(items, 0ms) > 0)
        {
            SPDLOG_TRACE("poll() was successful, receiving data...");
            KNP_UNROLL_LOOP()
            do
            {
                result = sub_socket_.recv(msg, zmq::recv_flags::dontwait);

                if (result.has_value())
                {
                    SPDLOG_TRACE("Endpoint received {} bytes.", result.value());
                }
                else
                {
                    SPDLOG_WARN("Endpoint receivied error [EAGAIN].");
                }
            } while (!result.has_value());
        }
        else
        {
            SPDLOG_DEBUG("poll() returned 0, exiting...");
            return std::nullopt;
        }
    }
    catch (const zmq::error_t &e)
    {
        SPDLOG_CRITICAL(e.what());
        throw;
    }

    return msg;
}
}  // namespace knp::core::messaging::impl
