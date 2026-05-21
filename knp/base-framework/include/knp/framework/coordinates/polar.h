/**
 * @file polar.h
 * @brief Implementations of radial coordinates.
 * @kaspersky_support Artiom N.
 * @date 22.03.2023
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

#include <boost/geometry.hpp>  // IWYU pragma: export


/**
 * @brief Namespace for the radial coordinates.
 */
namespace knp::framework::coordinates::radial
{

/**
 * @brief Namespace for polar coordinates.
 */
namespace d2
{

/**
 * @brief Predefined tag name for one coordinate.
 */
constexpr char coordinate_tag_name[]{"polar_coordinate"};
/**
 * @brief Predefined tag name for array of coordinates.
 */
constexpr char coordinate_array_tag_name[]{"polar_coordinates_array"};

/**
 * @brief Polar coordinates.
 * 
 * @tparam CoordinateType coordinate value type.
 * @tparam DegreeOrRadian coordinate angle type.
 */
template <
    typename CoordinateType, typename DegreeOrRadian = boost::geometry::degree,
    class CoordinateSystem = boost::geometry::cs::polar<DegreeOrRadian>>
class coordinate : public boost::geometry::model::point<CoordinateType, 2, CoordinateSystem>
{
public:
    /**
     * @brief Default constructor.
     */
    constexpr coordinate() = default;

    /**
     * @brief Constructor that uses radius and phi values.
     * 
     * @param radius radial distance.
     * @param phi polar angle.
     */
    constexpr coordinate(CoordinateType const& radius, CoordinateType const& phi)
        : boost::geometry::model::point<CoordinateType, 2, CoordinateSystem>(radius, phi)
    {
    }

    /**
     * @brief Get radius value.
     * 
     * @return radial distance.
     */
    constexpr CoordinateType const& radius() const { return this->template get<0>(); }

    /**
     * @brief Get phi value.
     * 
     * @return polar angle.
     */
    constexpr CoordinateType const& phi() const { return this->template get<1>(); }

    /**
     * @brief Set radius value.
     * 
     * @param v radius value.
     */
    void radius(CoordinateType const& v) { return this->template set<0>(v); }  // NOLINT

    /**
     * @brief Set phi value.
     * 
     * @param v phi value.
     */
    void phi(CoordinateType const& v) { return this->template set<1>(v); }  // NOLINT
};
}  // namespace d2


/**
 * @brief Namespace for spherical coordinates.
 */
namespace d3
{
/**
 * @brief Predefined tag name for one coordinate.
 */
constexpr char coordinate_tag_name[]{"spherical_coordinate"};
/**
 * @brief Predefined tag name for array of coordinates.
 */
constexpr char coordinate_array_tag_name[]{"spherical_coordinates_array"};

/**
 * @brief Spherical coordinates.
 * 
 * @tparam CoordinateType coordinate value type.
 * @tparam DegreeOrRadian coordinate angle type.
 */
template <
    typename CoordinateType, typename DegreeOrRadian = boost::geometry::degree,
    class CoordinateSystem = boost::geometry::cs::polar<DegreeOrRadian>>
class coordinate : public boost::geometry::model::point<CoordinateType, 3, CoordinateSystem>
{
public:
    /**
     * @brief Default constructor.
     */
    constexpr coordinate() = default;

    /**
     * @brief Constructor that uses radius, theta, and phi values.
     * 
     * @param radius radial distance.
     * @param theta polar angle.
     * @param phi azimuthal angle.
     */
    constexpr coordinate(CoordinateType const& radius, CoordinateType const& theta, CoordinateType const& phi)
        : boost::geometry::model::point<CoordinateType, 3, CoordinateSystem>(radius, theta, phi)
    {
    }

    /**
     * @brief Get radius value.
     * 
     * @return radial distance.
     */
    constexpr CoordinateType const& radius() const { return this->template get<0>(); }

    /**
     * @brief Get theta value.
     * 
     * @return polar angle.
     */
    constexpr CoordinateType const& theta() const { return this->template get<1>(); }

    /**
     * @brief Get phi value.
     * 
     * @return azimuthal angle.
     */
    constexpr CoordinateType const& phi() const { return this->template get<2>(); }

    /**
     * @brief Set radius value.
     * 
     * @param v radius value.
     */
    void radius(CoordinateType const& v) { return this->template set<0>(v); }  // NOLINT

    /**
     * @brief Set theta value.
     * 
     * @param v theta value.
     */
    void theta(CoordinateType const& v) { return this->template set<1>(v); }  // NOLINT

    /**
     * @brief Set phi value.
     * 
     * @param v phi value.
     */
    void phi(CoordinateType const& v) { return this->template set<2>(v); }  // NOLINT
};

}  // namespace d3

}  // namespace knp::framework::coordinates::radial
