/**
 * @file backend_loader.cpp
 * @brief Backend loader implementation.
 * @kaspersky_support Artiom N.
 * @date 16.03.2023
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

#include <knp/framework/backend_loader.h>

#include <spdlog/spdlog.h>

#include <functional>

#include <boost/dll/import.hpp>


namespace knp::framework
{

std::function<BackendLoader::BackendCreateFunction> BackendLoader::make_creator(
    const std::filesystem::path &backend_path)
{
    auto creator_iter = creators_.find(backend_path.string());

    if (creator_iter != creators_.end())
    {
        return creator_iter->second;
    }

    SPDLOG_INFO("Loading backend by path \"{}\"...", backend_path.string());

    auto creator = boost::dll::import_alias<BackendCreateFunction>(
        boost::filesystem::path(backend_path), "create_knp_backend", boost::dll::load_mode::append_decorations);

    SPDLOG_DEBUG("Created backend creator.");

    creators_[backend_path.string()] = creator;

    return creator;
}


std::shared_ptr<core::Backend> BackendLoader::load(const std::filesystem::path &backend_path)
{
    auto creator = make_creator(backend_path);
    SPDLOG_INFO("Created backend instance.");
    std::shared_ptr<core::Backend> result = creator();
    if (!result)
    {
        SPDLOG_ERROR("Unable to load backend from " + backend_path.string());
        throw std::runtime_error("Couldn't load backend from path \"" + backend_path.string() + "\"");
    }
    return result;
}


bool BackendLoader::is_backend(const std::filesystem::path &backend_path)
{
    SPDLOG_INFO("Checking library by path \"{}\"...", backend_path.string());
    const boost::dll::shared_library lib{
        boost::filesystem::path(backend_path), boost::dll::load_mode::append_decorations};
    return lib.has("create_knp_backend");
}

}  // namespace knp::framework
