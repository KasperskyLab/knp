/**
 * @file data_read.cpp
 * @brief Reading from dataset.
 * @kaspersky_support A. Vartenkov
 * @date 30.08.2024
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

#include "data_read.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <utility>

/**
 * @brief Read buffers from binary data file.
 * @param path_to_data path to binary data file.
 */
std::vector<std::vector<unsigned char>> read_images_from_file(const std::string &path_to_data)
{
    std::ifstream file_stream(path_to_data, std::ios::binary);
    std::vector<unsigned char> buffer;
    std::vector<std::vector<unsigned char>> result;
    while (file_stream.good())
    {
        buffer.resize(input_size);
        file_stream.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(input_size));
        result.push_back(std::move(buffer));
        buffer.clear();
    }
    return result;
}


std::vector<std::vector<bool>> image_to_spikes(const std::vector<unsigned char> &image, std::vector<double> &state)
{
    std::vector<std::vector<bool>> ret;
    int i;
    for (i = 0; i < intensity_levels; ++i)
    {
        std::vector<bool> spikes(input_size, false);
        for (int l = 0; l < input_size; ++l)
        {
            state[l] += state_increment_factor * image[l];
            if (state[l] >= 1.)
            {
                spikes[l] = true;
                --state[l];
            }
        }
        ret.push_back(spikes);
    }
    for (; i < frames_per_image; ++i) ret.push_back(std::vector<bool>(input_size, false));
    return ret;
}


// Read image dataset from a binary file and trasnform it into a vector of boolean frames.
std::vector<std::vector<bool>> read_spike_frames(const std::filesystem::path &path_to_data)
{
    auto images = read_images_from_file(path_to_data);
    std::vector<std::vector<bool>> result;
    std::vector<double> state(input_size, 0.);
    for (size_t img_num = 0; img_num < images.size(); ++img_num)
    {
        std::vector<std::vector<bool>> spikes_per_image = image_to_spikes(images[img_num], state);
        std::transform(
            spikes_per_image.begin(), spikes_per_image.end(), std::back_inserter(result),
            [](auto &v) { return std::move(v); });
    }
    return result;
}


Labels read_labels(const std::filesystem::path &classes_file, int learning_period, int offset)
{
    std::ifstream file_stream(classes_file);
    Labels labels;
    int cla;
    while (file_stream.good())
    {
        std::string str;
        if (!std::getline(file_stream, str).good()) break;
        if (offset > 0)
        {
            --offset;
            continue;
        }
        std::stringstream ss(str);
        ss >> cla;
        std::vector<bool> buffer(input_size, false);
        buffer[cla] = true;
        if (labels.train_.size() >= learning_period) labels.test_.push_back(cla);
        for (int i = 0; i < frames_per_image; ++i) labels.train_.push_back(buffer);
    }
    return labels;
}
