/**
 * @file backend_loader_test.cpp
 * @brief Backend loading testing.
 * @kaspersky_support Artiom N.
 * @date 17.03.2023
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

#include <knp/framework/backend_loader.h>

#include <tests_common.h>

#include <filesystem>


TEST(FrameworkSuite, StBackendLoaderLoad)
{
    knp::framework::BackendLoader backend_loader;
    auto cpu_st_backend{backend_loader.load(knp::testing::get_backend_path())};

    EXPECT_NO_THROW((void)cpu_st_backend->get_uid());  //!OCLINT(False positive)
}


TEST(FrameworkSuite, MtBackendLoaderLoad)
{
    knp::framework::BackendLoader backend_loader;
    auto cpu_st_backend{backend_loader.load(knp::testing::get_backend_path("knp-cpu-multi-threaded-backend"))};

    EXPECT_NO_THROW((void)cpu_st_backend->get_uid());  //!OCLINT(False positive)
}


TEST(FrameworkSuite, BackendLoaderCheck)
{
    const knp::framework::BackendLoader backend_loader;

    ASSERT_TRUE(backend_loader.is_backend(knp::testing::get_backend_path()));
}


TEST(FrameworkSuite, BackendLoaderExceptionCheck)
{
    knp::framework::BackendLoader backend_loader;
    ASSERT_ANY_THROW(backend_loader.load("Almost certainly a wrong path"));
}


TEST(FrameworkSuite, BackendGetDevices)  //!OCLINT(False positive)
{
    knp::framework::BackendLoader backend_loader;
    auto cpu_st_backend = backend_loader.load(knp::testing::get_backend_path());

    const std::set<knp::core::UID> dev_uids{knp::core::UID()};

    ASSERT_THROW(cpu_st_backend->select_devices(dev_uids), std::logic_error);  //!OCLINT(False positive)
}
