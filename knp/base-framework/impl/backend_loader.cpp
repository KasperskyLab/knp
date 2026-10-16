/**
 * @file backend_loader.cpp
 * @brief Backend loader implementation.
 * @kaspersky_support Artiom N.
 * @date 16.03.2023
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

#include <knp/framework/backend_loader.h>

#include <spdlog/spdlog.h>

#include <functional>

#include <boost/dll.hpp>
#include <boost/dll/import.hpp>
#include <boost/exception/all.hpp>


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

    SPDLOG_DEBUG("Trying to load backend by path \"{}\"...", backend_path.string());

    auto creator = boost::dll::import_alias<BackendCreateFunction>(
        boost::filesystem::path(backend_path), "create_knp_backend", boost::dll::load_mode::append_decorations);

    SPDLOG_DEBUG("Created backend creator.");

    creators_[backend_path.string()] = creator;

    return creator;
}


std::shared_ptr<core::Backend> BackendLoader::load_backend_library(const std::filesystem::path &backend_path)
{
    try
    {
        if (!is_backend(backend_path)) return nullptr;

        auto creator = make_creator(backend_path);

        SPDLOG_DEBUG("Created backend instance from {}", backend_path.c_str());

        std::shared_ptr<core::Backend> result = creator();

        if (result)
        {
            SPDLOG_INFO("Backend loaded.");
            return result;
        }
    }
    catch (const boost::system::system_error &e)
    {
        SPDLOG_DEBUG("Loading error: {}", e.code().message().c_str());
    }
    catch (const boost::exception &e)
    {
        SPDLOG_DEBUG("Loading error: {}", boost::diagnostic_information(e));
    }
    catch (const std::exception &e)
    {
        SPDLOG_DEBUG("Loading error: {}", e.what());
    }

    return nullptr;
}


std::shared_ptr<core::Backend> BackendLoader::load_by_name(
    const std::string &backend_name, const std::vector<std::filesystem::path> &add_paths)
{
    std::vector<std::string> backend_names = {backend_name};

    if (0 != backend_name.rfind("knp-", 0))
    {
        const std::string prefixed_bn = "knp-" + backend_name;
        backend_names.push_back(prefixed_bn);
        SPDLOG_TRACE("Adding \"{}\" to backend names...", prefixed_bn.c_str());
    }

    for (const auto &lb_name : backend_names)
    {
        SPDLOG_TRACE("Trying to load \"{}\" from custom paths...", lb_name.c_str());
        for (const auto &backend_path : add_paths)
        {
            auto backend_lib_path = backend_path / lb_name;
            auto backend = load_backend_library(backend_lib_path);

            if (backend != nullptr)
            {
                return backend;
            }
        }

        static const std::vector<std::filesystem::path> paths = {
            boost::dll::program_location().parent_path().c_str(),
            std::filesystem::current_path(),
#if !defined(WIN32)
            "/usr/lib/",
            "/usr/lib32/",
            "/usr/lib64/",
            "/usr/bin/",
            "/usr/local/lib/",
            "/usr/local/lib32/",
            "/usr/local/lib64/",
            "/usr/local/bin/",
#else
        // For OS Windows.
#endif
        };

        SPDLOG_TRACE("Trying to load \"{}\" from standard paths...", lb_name.c_str());
        for (const auto &backend_path : paths)
        {
            auto backend_lib_path = backend_path / lb_name;
            auto backend = load_backend_library(backend_path);

            if (backend != nullptr)
            {
                return backend;
            }
        }
    }

    SPDLOG_ERROR("Unable to load backend.");
    throw std::runtime_error("Couldn't load backend \"" + backend_name + "\"");
}


std::shared_ptr<core::Backend> BackendLoader::load(const std::filesystem::path &backend_path)
{
    auto result = load_backend_library(backend_path);

    if (!result)
    {
        SPDLOG_ERROR("Unable to load backend from " + backend_path.string());
        throw std::runtime_error("Couldn't load backend from path \"" + backend_path.string() + "\"");
    }

    SPDLOG_INFO("Backend loaded.");
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
