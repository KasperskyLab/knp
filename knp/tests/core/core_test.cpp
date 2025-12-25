/**
 * @file core_test.cpp.
 * @brief UUID tests.
 * @kaspersky_support Artiom N.
 * @date 21.08.2023.
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

#include <knp/core/core.h>

#include <tests_common.h>


namespace knp::testing
{

TEST(CoreSuite, TagMapTest)
{
    knp::core::TagMap tag_map;

    tag_map.get_tag("test") = std::string("test_val");

    ASSERT_EQ(tag_map.get_tag<std::string>("test"), "test_val");

    tag_map["test"] = std::string("new");

    ASSERT_EQ(tag_map.get_tag<std::string>("test"), "new");

    ASSERT_TRUE(!tag_map.empty());
}

}  // namespace knp::testing
