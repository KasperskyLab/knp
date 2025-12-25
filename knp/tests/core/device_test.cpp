/**
 * @file device_test.cpp
 * @brief Device tests.
 * @kaspersky_support Artiom N.
 * @date 09.08.2024
 * @license Apache 2.0
 * @copyright © 2024-2025 AO Kaspersky Lab
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

#if !defined(_MSC_VER)
extern "C"
{
#    include <unistd.h>
}
#endif

#include <knp/backends/cpu-single-threaded/backend.h>
#include <knp/devices/cpu.h>

#include <tests_common.h>

namespace knp::testing
{

TEST(DeviceTestSuite, CPUTest)
{
    GTEST_SKIP() << "This test doesn't work under Github builders";

#if defined(WIN32)
    GTEST_SKIP() << "Test needs driver under Windows";
#else
    if (geteuid() != 0)
    {
        GTEST_SKIP() << "This test must be run under root.";
    }
#endif

    auto processors = knp::devices::cpu::list_processors();

    ASSERT_GE(processors.size(), 1);

    for (auto &device : processors) SPDLOG_DEBUG("CPU name: {}.", device.get_name());
}


TEST(DeviceTestSuite, BackendDevicesTest)  // cppcheck-suppress syntaxError
{
    GTEST_SKIP() << "This test doesn't work under Github builders";

#if defined(WIN32)
    GTEST_SKIP() << "Test needs driver under Windows";
#else
    if (geteuid() != 0)
    {
        SPDLOG_WARN("This test must be run under root.");
        GTEST_SKIP() << "This test must be run under root.";
    }
#endif

    knp::backends::single_threaded_cpu::SingleThreadedCPUBackend backend;

    auto &devices = backend.get_current_devices();

    std::set<knp::core::UID> dus;

    backend.select_devices(dus);

    for (auto &d : devices)
    {
        std::cout << d->get_name() << std::endl;
    }

    // std::cout << device.get_name() << std::endl;
}

}  // namespace knp::testing
