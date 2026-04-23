/**
 * @file cartesian.h
 * @brief Representation of Cartesian coordinates.
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

#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/point_xyz.hpp>


/**
 * @brief Namespace for Cartesian coordinates.
 */
namespace knp::framework::coordinates::cartesian
{

/**
 * @brief Namespace for two-dimensional Cartesian coordinates.
 */
namespace d2
{
/**
 * @brief Predefined tag name for one coordinate.
 */
constexpr char coordinate_tag_name[]{"ortho_coordinate_2d"};
/**
 * @brief Predefined tag name for array of coordinates.
 */
constexpr char coordinate_array_tag_name[]{"ortho_coordinates_2d_array"};

/**
 * @brief 2D point coordinates.
 * 
 * @tparam CoordinateType coordinate value type.
 */
template <typename CoordinateType>
using coordinate = boost::geometry::model::d2::point_xy<CoordinateType>;
}  // namespace d2


/**
 * @brief Namespace for three-dimensional Cartesian coordinates.
 */
namespace d3
{
/**
 * @brief Predefined tag name for one coordinate.
 */
constexpr char coordinate_tag_name[]{"ortho_coordinate_3d"};
/**
 * @brief Predefined tag name for array of coordinates.
 */
constexpr char coordinate_array_tag_name[]{"ortho_coordinates_3d_array"};

/**
 * @brief 3D point coordinates.
 * 
 * @tparam CoordinateType coordinate value type.
 */
template <typename CoordinateType>
using coordinate = boost::geometry::model::d3::point_xyz<CoordinateType>;
}  // namespace d3

}  // namespace knp::framework::coordinates::cartesian
