/**
 * @file csv_content.h
 * @brief CSV helper file.
 * @kaspersky_support An. Vartenkov
 * @date 22.03.2024
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

#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


#pragma once

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wconditional-uninitialized"
#elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include <csv2/reader.hpp>
#include <csv2/writer.hpp>

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif


namespace knp::framework::sonata
{
namespace fs = std::filesystem;


class CsvContent
{
public:
    const auto &get_header() const { return header_; }

    void set_header(std::vector<std::string> header)
    {
        header_ = std::move(header);
        for (size_t i = 0; i < header_.size(); ++i) header_index_.emplace(header_[i], i);
    }

    void add_row(std::vector<std::string> new_row) { values_.push_back(std::move(new_row)); }

    std::vector<std::string> get_row(size_t row_n) { return values_[row_n]; }

    template <class V>
    V get_value(size_t row, const std::string &col) const;

    auto get_rc_size() const { return std::make_pair(values_.size(), header_.size()); }

private:
    std::vector<std::string> header_;
    std::unordered_map<std::string, size_t> header_index_;
    std::vector<std::vector<std::string>> values_;
};

CsvContent load_csv_content(const fs::path &csv_path);

void save_csv_content(const CsvContent &csv_data, const fs::path &csv_path);

const std::vector<std::string> edge_file_header = {"edge_type_id", "dynamics_params", "model_template"};
const std::vector<std::string> node_file_header = {"node_type_id", "model_type", "model_template", "model_name"};

}  // namespace knp::framework::sonata
