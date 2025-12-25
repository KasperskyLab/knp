/**
 * @file uid_test.cpp
 * @brief UUID tests.
 * @kaspersky_support Artiom N.
 * @date 13.07.2023.
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

#include <knp/core/uid.h>

#include <tests_common.h>


namespace knp::testing
{

TEST(UidSuite, UidGenerator)
{
    using uuid = ::boost::uuids::uuid;

    auto gen = ::knp::core::continuously_uid_generator();
    gen.reset();
    auto uid = gen();

    ASSERT_EQ(uuid{{1}}, uid);
    ASSERT_NE(uuid{{2}}, uid);

    auto u_gen = ::knp::core::continuously_uid_generator();

    for (uint8_t i = 2; i < 100; ++i) ASSERT_EQ(uuid{{i}}, u_gen());
}


TEST(UidSuite, UidToString)  // cppcheck-suppress syntaxError
{
    ::knp::core::UID uid{::boost::uuids::uuid{{1, 2, 3}}};

    ASSERT_EQ("01020300-0000-0000-0000-000000000000", std::string(uid));

    std::cout << uid << std::endl;
}


TEST(UidSuite, UidFromStream)
{
    // ::knp::core::UID uid;

    // std::istringstream ss;

    // ss << "01020300-0000-0000-0000-000000000000";
    // uid << ss;

    // ASSERT_EQ(uid, ::knp::core::UID(::boost::uuids::uuid{{1, 2, 3}}));
}


TEST(UidSuite, UidEq)
{
    ::knp::core::UID uid1{::boost::uuids::uuid{{1, 2, 3}}};
    ::knp::core::UID uid2{::boost::uuids::uuid{{1, 2, 3}}};
    ::knp::core::UID uid3{::boost::uuids::uuid{{3, 2, 1}}};
    ::knp::core::UID uid4(uid1);

    ASSERT_EQ(uid1, uid2);
    ASSERT_NE(uid1, uid3);
    ASSERT_EQ(uid4, uid2);
    ASSERT_LT(uid3, uid1);
}


TEST(UidSuite, UidHash)
{
    ::knp::core::UID uid1{::boost::uuids::uuid{{1, 2, 3}}};

    std::unordered_map<knp::core::UID, knp::core::UID, knp::core::uid_hash> uid_container;

    uid_container[uid1] = uid1;
    ASSERT_EQ(uid_container[uid1], uid1);
}

}  // namespace knp::testing
