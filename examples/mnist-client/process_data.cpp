/**
 * @file process_data.cpp
 * @brief Functions to process a binary file.
 * @date 13.09.2024
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

#include "process_data.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


/**
 * @brief Filesystem namespace alias.
 */
namespace fs = std::filesystem;


/**
 * @brief Visualize data instances.
 * @param data vector of flattened images.
 * @param size output image size.
 */
cv::Mat draw_data(const std::vector<unsigned char> &data, const cv::Size &size)
{
    cv::Mat out_img(size, CV_8UC1);
    for (int y = 0; y < size.height; ++y)
    {
        for (int x = 0; x < size.width; ++x)
        {
            out_img.at<unsigned char>(cv::Point2i{x, y}) = data[size.width * y + x];
        }
    }
    cv::imshow("Data", out_img);
    std::cout << cv::sum(out_img) << std::endl;
    cv::waitKey(0);
    return out_img;
}


/**
 * @brief Turn data frames into spikes.
 * @param buf data frame.
 * @param num_levels number of intensity levels that the original interval of `0 - 255` gets divided into.
 * @param num_frames all number of steps per data instance.
 */
std::vector<std::vector<bool>> image_to_spikes(std::vector<unsigned char> &buf, int num_levels, int num_frames_all)
{
    const double delta = 256.0 / num_levels;
    std::vector<std::vector<bool>> result(num_frames_all);
    for (size_t frame_num = 0; frame_num < std::min(num_frames_all, num_levels); ++frame_num)
    {
        result[frame_num].resize(buf.size());
        for (size_t input_pin = 0; input_pin < buf.size(); ++input_pin)
        {
            result[frame_num][input_pin] = (buf[input_pin] >= delta * static_cast<double>(frame_num + 1));
        }
    }
    return result;
}


/**
 * @brief Read buffers from binary data file.
 * @param path_to_data path to binary data file.
 * @param input_size data buffer length in bytes.
 */
std::vector<std::vector<unsigned char>> read_images_from_file(const fs::path &path_to_data, size_t input_size)
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


/**
 * @brief Convert binary files to boolean vectors, one for each step.
 * @note The function only works for small datasets, such as MNIST. Dataset is considered small if it fits into RAM.
 * @param path_to_data path to binary data file.
 * @param input_size size of input buffer.
 * @param frames_per_image number of network steps per one image.
 * @param intensity_levels number of intensity levels.
 * @param skip number of images to skip.
 * @return vector of boolean frames, one per each network step.
 * @see `image_to_spikes` function.
 */
std::vector<std::vector<bool>> read_spikes_from_grayscale_file(
    const fs::path &path_to_data, size_t input_size, int frames_per_image, int intensity_levels, size_t skip)
{
    auto images = read_images_from_file(path_to_data, input_size);
    std::vector<std::vector<bool>> result;
    for (size_t img_num = skip; img_num < images.size(); ++img_num)
    {
        std::vector<std::vector<bool>> spikes_per_image =
            image_to_spikes(images[img_num], intensity_levels, frames_per_image);
        std::transform(
            spikes_per_image.begin(), spikes_per_image.end(), std::back_inserter(result),
            [](auto &v) { return std::move(v); });
    }
    return result;
}
