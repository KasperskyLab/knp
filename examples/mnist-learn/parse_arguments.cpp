/**
 * @file parse_arguments.cpp
 * @brief Parsing of command line arguments.
 * @kaspersky_support D. Postnikov
 * @date 03.02.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include "parse_arguments.h"

#include <iostream>
#include <string>

#include <boost/program_options.hpp>


namespace po = boost::program_options;

std::optional<ModelDescription> parse_arguments(int argc, char** argv)
{
    po::options_description desc("Usage");
    desc.add_options()("help,h", "print available options")(
        "model,m", po::value<std::string>()->default_value("blifat"), "model type. allowed options are: blifat, altai")(
        "train_iters,t", po::value<size_t>()->default_value(60000), "amount of images for training")(
        "inference_iters,i", po::value<size_t>()->default_value(10000), "amount of images for inference")(
        "images", po::value<std::string>()->default_value("MNIST.bin"), "path to raw images file")(
        "labels", po::value<std::string>()->default_value("MNIST.target"), "path to images labels file")(
        "backend,b", po::value<std::string>()->default_value("knp-cpu-single-threaded-backend"), "path to backend")(
        "log_path", po::value<std::string>()->default_value(""),
        "path for putting logs. if no path is specified, no logs will be produced.")(
        "model_path", po::value<std::string>()->default_value(""),
        "path for saving trained model. if no path is specified, model wont be saved.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        return std::nullopt;
    }

    ModelDescription model_desc;

    if (vm.count("model"))
    {
        std::string model_type = vm["model"].as<std::string>();
        if (model_type == "blifat")
        {
            model_desc.type_ = SupportedModelType::BLIFAT;
        }
        else if (model_type == "altai")
        {
            model_desc.type_ = SupportedModelType::AltAI;
        }
        else
        {
            std::cout << "Not supported model type." << std::endl;
            std::cout << desc << std::endl;
            return std::nullopt;
        }
    }
    else
    {
        std::cout << "Model type not specified." << std::endl;
        std::cout << desc << std::endl;
        return std::nullopt;
    }

    if (vm.count("train_iters"))
    {
        model_desc.train_images_amount_ = vm["train_iters"].as<size_t>();
    }
    else
    {
        std::cout << "Train iterations not specified." << std::endl;
        std::cout << desc << std::endl;
        return std::nullopt;
    }

    if (vm.count("inference_iters"))
    {
        model_desc.inference_images_amount_ = vm["inference_iters"].as<size_t>();
    }
    else
    {
        std::cout << "Inference iterations not specified." << std::endl;
        std::cout << desc << std::endl;
        return std::nullopt;
    }

    if (vm.count("images"))
    {
        model_desc.images_file_path_ = vm["images"].as<std::string>();
    }
    else
    {
        std::cout << "Images path not specified." << std::endl;
        std::cout << desc << std::endl;
        return std::nullopt;
    }

    if (vm.count("labels"))
    {
        model_desc.labels_file_path_ = vm["labels"].as<std::string>();
    }
    else
    {
        std::cout << "Labels path not specified." << std::endl;
        std::cout << desc << std::endl;
        return std::nullopt;
    }

    if (vm.count("backend"))
    {
        model_desc.backend_path_ = vm["backend"].as<std::string>();
    }
    else
    {
        std::cout << "Backend path not specified." << std::endl;
        std::cout << desc << std::endl;
        return std::nullopt;
    }

    if (vm.count("log_path"))
    {
        model_desc.log_path_ = vm["log_path"].as<std::string>();
    }
    else
    {
        model_desc.log_path_ = "";
    }

    if (vm.count("model_path"))
    {
        model_desc.model_saving_path_ = vm["model_path"].as<std::string>();
    }
    else
    {
        model_desc.model_saving_path_ = "";
    }

    return model_desc;
}
