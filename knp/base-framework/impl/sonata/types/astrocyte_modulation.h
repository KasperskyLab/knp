/**
 * @file astrocyte_modulation.h
 * @brief SONATA serialization of astrocyte-controlled projections and original synapse weights.
 * @author Yuliya Tsybina
 * @date 2026
 * @license Apache 2.0
 * @copyright © 2026 Lobachevsky University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <knp/astrocyte-traits/type_traits.h>
#include <knp/core/tag.h>
#include <knp/core/uid.h>

#include <string>
#include <utility>
#include <vector>

#include <boost/lexical_cast.hpp>

#include "../highfive.h"

namespace knp::framework::sonata
{

/**
 * @brief Save controlled projection UIDs and original weights from an astrocyte population.
 * @param tags population tags.
 * @param group population HDF5 group.
 */
inline void save_astrocyte_modulation(const core::TagMap &tags, HighFive::Group &group)
{
    if (!tags.exists(astrocyte_traits::astrocyte_modulated_projections_tag)) return;
    auto modulation = group.createGroup("astrocyte_modulation");
    const auto &uids = tags.get_tag<std::vector<core::UID>>(astrocyte_traits::astrocyte_modulated_projections_tag);
    std::vector<std::string> names;
    names.reserve(uids.size());
    for (const auto &uid : uids) names.push_back(std::string(uid));
    if (!names.empty()) modulation.createDataSet("projection_uids", names);
    if (tags.exists(astrocyte_traits::astrocyte_original_weights_tag))
    {
        auto weights_group = modulation.createGroup("original_weights");
        const auto &weights = tags.get_tag<astrocyte_traits::AstrocyteProjectionWeights>(
            astrocyte_traits::astrocyte_original_weights_tag);
        for (const auto &name : names)
        {
            const auto found = weights.find(name);
            if (found != weights.end()) weights_group.createDataSet(name, found->second);
        }
    }
}


/**
 * @brief Restore astrocyte-controlled projections and original weights if present.
 * @param group population HDF5 group.
 * @param tags population tags to restore.
 */
inline void load_astrocyte_modulation(const HighFive::Group &group, core::TagMap &tags)
{
    if (!group.exist("astrocyte_modulation")) return;
    const auto modulation = group.getGroup("astrocyte_modulation");
    std::vector<core::UID> uids;
    if (modulation.exist("projection_uids"))
    {
        for (const auto &name : modulation.getDataSet("projection_uids").read<std::vector<std::string>>())
            uids.emplace_back(boost::lexical_cast<boost::uuids::uuid>(name));
    }
    tags[astrocyte_traits::astrocyte_modulated_projections_tag] = std::move(uids);
    if (modulation.exist("original_weights"))
    {
        const auto weights_group = modulation.getGroup("original_weights");
        astrocyte_traits::AstrocyteProjectionWeights weights;
        for (const auto &name : weights_group.listObjectNames())
            weights[name] = weights_group.getDataSet(name).read<std::vector<float>>();
        tags[astrocyte_traits::astrocyte_original_weights_tag] = std::move(weights);
    }
}

}  // namespace knp::framework::sonata
