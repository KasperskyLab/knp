/**
 * @file backend_loader.h
 * @brief Backend loader interface.
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

#pragma once

#include <knp/core/backend.h>
#include <knp/core/impexp.h>

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{

/**
 * @brief The BackendLoader class is a definition of a backend loader.
 */
class KNP_DECLSPEC BackendLoader
{
public:
    /**
     * @brief Load backend.
     * @param backend_path path to backend.
     * @return shared pointer to Backend object.
     */
    std::shared_ptr<core::Backend> load(const std::filesystem::path &backend_path);
    /**
     * @brief Check if the specified path points to a backend.
     * @param backend_path path to backend.
     * @return `true` if the library pointed by `backend_path` is a backend.
     */
    static bool is_backend(const std::filesystem::path &backend_path);

public:
    /**
     * @brief Function type that returns a backend instance.
     * @return shared pointer to Backend object.
     */
    typedef std::shared_ptr<core::Backend>(BackendCreateFunction)();

protected:
    /**
     * @brief Create a backend loader that will load and create a backend instance.
     * @param backend_path path to backend.
     * @return pointer to the backend creation function.
     */
    std::function<BackendCreateFunction> make_creator(const std::filesystem::path &backend_path);

private:
    // std::filesystem::path doesn't work on any compilers.
    std::unordered_map<std::string, std::function<BackendCreateFunction>> creators_;
};

}  // namespace knp::framework
