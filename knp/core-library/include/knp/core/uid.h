/**
 * @file uid.h
 * @brief UID class and routines.
 * @kaspersky_support Artiom N.
 * @date 01.02.2021
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

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <sstream>
#include <string>
#include <utility>

#include <boost/functional/hash.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>


/**
 * @brief Core library namespace.
 */
namespace knp::core
{

#if defined(_DEBUG) || defined(DEBUG)
#    define _ENABLE_PSEUDO_UID_GENERATOR 1
#endif

/**
 * @brief UID generator for entities.
 * 
 * @details You can use this class for debugging. UID is displayed as a 128-bit number.
 */
class continuously_uid_generator
{
public:
    /**
     * @brief Generate next UID.
     * 
     * @return Boost UUID object.
     */
    ::boost::uuids::uuid operator()() const;
    /**
     * @brief Reset UID counter.
     * 
     * @param initial_value UID value from which to restart the counter.
     */
    static void reset(uint64_t initial_value = 1);
};

#if defined(_ENABLE_PSEUDO_UID_GENERATOR)
#    define uid_generator continuously_uid_generator
#else
/**
 * @brief Default UID generator.
 */
#    define uid_generator ::boost::uuids::random_generator
#endif


/**
 * @brief The UID class is a definition of unique identifiers for entities.
 */
struct UID
{
    /**
     * @brief Generate a random UID.
     * 
     * @param random boolean value that takes the values `true` or `false`.
     *        If `true`, the constructor generates a random UID. If `false`, the constructor generates a null UID.
     * 
     * @todo Optimize it.
     */
    explicit UID(bool random) : tag(random ? uid_generator()() : ::boost::uuids::nil_uuid()) {}

    /**
     * @brief Default UID constructor.
     */
    UID() : UID(true) {}

    /**
     * @brief Create a UID from `boost::uuids::uuid`.
     * 
     * @param guid constant value to copy to UID.
     */
    explicit UID(const ::boost::uuids::uuid &guid) : tag(guid) {}
    /**
     * @brief Create a UID from `boost::uuids::uuid`.
     * 
     * @param guid value to store as a UID.
     */
    explicit UID(::boost::uuids::uuid &&guid) : tag(guid) {}

    /**
     * @brief Create a UID from an array of 16 uint8_t integers.
     * 
     * @param guid_value value to store as a UID.
     */
    explicit UID(const std::array<
                 boost::uuids::uuid::value_type, sizeof(boost::uuids::uuid::data) / sizeof(boost::uuids::uuid::data[0])>
                     &guid_value)
    {
        std::copy(guid_value.begin(), guid_value.end(), tag.begin());
    }

    /**
     * @brief Constructor that creates a copy of an existing UID.
     * 
     * @details The constructor copies all attribute values of the existing UID to the new UID.
     */
    UID(const UID &) = default;

    /**
     * @brief Convert UID to `boost::uuids::uuid`.
     * 
     * @return stored UID value.
     * 
     * @details Constant method.
     */
    operator const ::boost::uuids::uuid &() const { return tag; }

    /**
     * @brief Convert UID to `boost::uuids::uuid`.
     * 
     * @return stored UID value.
     */
    operator ::boost::uuids::uuid &() { return tag; }

    /**
     * @brief Convert UID to string.
     * 
     * @return string representation of the UID.
     */
    explicit operator ::std::string() const
    {
        std::stringstream ss;
        ss << tag;
        return ss.str();
    }

    /**
     * @brief Check if UID is valid.
     * 
     * @return `true` if UID is a non-zero value.
     */
    explicit operator bool() const { return !tag.is_nil(); }

    /**
     * @brief Check if the current UID is less than the specified UID.
     * 
     * @param uid UID to compare to the current UID.
     * 
     * @return `true` if the current UID is less than the specified UID.
     */
    bool operator<(const UID &uid) const { return uid.tag < tag; }

    /**
     * @brief Check if two UIDs are the same.
     * 
     * @param uid UID to compare to the current UID.
     * 
     * @return `true` if both UIDs are the same.
     */
    bool operator==(const UID &uid) const { return uid.tag == tag; }

    /**
     * @brief Check if two UIDs are different.
     * 
     * @param uid UID to compare to the current UID.
     * 
     * @return `true` if the UIDs are different.
     */
    bool operator!=(const UID &uid) const { return uid.tag != tag; }

    /**
     * @brief UID value.
     */
    ::boost::uuids::uuid tag;
};


/**
 * @brief Send UID to an output stream.
 * 
 * @param s output stream.
 * @param uid UID to send to the output stream.
 * 
 * @return modified stream.
 */
inline ::std::ostream &operator<<(std::ostream &s, const UID &uid)
{
    s << uid.tag;
    return s;
}

/**
 * @brief Get UID from an input stream.
 * 
 * @param s input stream.
 * @param uid UID to get from the input stream.
 * 
 * @return modified stream.
 */
inline ::std::istream &operator>>(std::istream &s, UID &uid)
{
    s >> uid.tag;
    return s;
}


/**
 * @brief UID hash functor type.
 */
struct uid_hash
{
    /**
     * @brief Get a hash value of the specified UID.
     * 
     * @param uid UID to convert to a hash value.
     * 
     * @return UID hash value.
     */
    auto operator()(const UID &uid) const { return boost::hash<boost::uuids::uuid>()(uid.tag); }
};

}  // namespace knp::core
