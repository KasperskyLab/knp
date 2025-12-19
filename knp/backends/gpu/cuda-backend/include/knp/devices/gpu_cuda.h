/**
 * @file gpu_cuda.h
 * @brief Class definition for CUDA device.
 * @kaspersky_support Artiom N.
 * @date 24.02.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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

#include <knp/core/device.h>
#include <knp/core/impexp.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <memory>
#include <string>
#include <vector>


/**
 * @namespace knp::devices
 * @brief Namespace for implementation of concrete devices.
 */

/**
 * @brief GPU device namespace.
 */
namespace knp::devices::gpu
{

/**
 * @brief The CUDA class is a definition of an interface to the CUDA device.
 */
class KNP_DECLSPEC CUDA final : public knp::core::Device  // cppcheck-suppress class_X_Y
{
public:
    /**
     * @brief Avoid copy of a CUDA device.
     */
    CUDA(const CUDA &) = delete;

    /**
     * @brief Avoid copy assignment of a CUDA device.
     */
    CUDA &operator=(const CUDA &) = delete;

    /**
     * @brief CUDA device move constructor.
     */
    CUDA(CUDA &&);

    /**
     * @brief CUDA device move operator.
     * @return reference to CUDA instance.
     */
    CUDA &operator=(CUDA &&) noexcept;

    /**
     * @brief CUDA device destructor.
     */
    ~CUDA() override;

public:
    /**
     * @brief Get device type.
     * @return device type.
     */
    [[nodiscard]] knp::core::DeviceType get_type() const override;

    /**
     * @brief Get device name.
     * @return device name in the arbitrary format.
     */
    [[nodiscard]] const std::string get_name() const override;

    /**
     * @brief Get CUDA device socket number.
     * @return socket number.
     */
    [[nodiscard]] uint32_t get_socket_number() const;

    /**
     * @brief Get power consumption details for the device in mWatts.
     * @return amount of consumed power.
     */
    [[nodiscard]] float get_power() const override;

    /**
     * @brief Get warp size from GPU.
     * @return Device warp size.
     */
    [[nodiscard]] unsigned int get_warp_size() const;

    /**
     * @brief Get number of multiprocessors.
     * @return number of multiprocessors.
     */
    [[nodiscard]] unsigned int get_mp_count() const;

    /**
     * @brief Get number of threads per multiprocessor.
     * @return number of threads per multiprocessor.
     */
    [[nodiscard]] unsigned int get_threads_per_mp() const;

    /**
     * @brief Get number of threads per block from GPU.
     * @return maximum number of threads.
     */
    [[nodiscard]] unsigned int get_max_threads_count() const;

    /**
     * @brief Get number of kernels  that GPU can possibly execute concurrently.
     * @return number of the concurrent kernels.
     */
    [[nodiscard]] unsigned int get_concurrent_kernels() const;

    /**
     * @brief Get maximum size of each dimension of a block from GPU.
     * @return dimensions sizes.
     */
    [[nodiscard]] const std::array<int, 3> get_max_threads_dim() const;

    /**
     * @brief Get maximum size of each dimension of a grid from GPU.
     * @return grid dimensions.
     */
    [[nodiscard]] const std::array<int, 3> get_max_grid_size() const;

    /**
     * @brief Get global memory available on device in bytes.
     * @return number of bytes available in the global memory.
     */
    [[nodiscard]] unsigned int get_global_memory_bytes() const;

    /**
     * @brief Get constant memory available on device in bytes.
     * @return number of bytes available in the constant memory.
     */
    [[nodiscard]] unsigned int get_constant_memory_bytes() const;

public:
    /**
     * @brief Get description of sGPU.
     * @return device properties structure.
     */
    [[nodiscard]] [[deprecated("Use methods")]] cudaDeviceProp get_device_prop() const;

private:
    /**
     * @brief CUDA device constructor.
     */
    explicit CUDA(uint32_t gpu_num);
    friend KNP_DECLSPEC std::vector<CUDA> list_cuda_processors();

private:
    uint32_t gpu_num_;
    cudaDeviceProp properties_;
};


/**
 * @brief List all GPUs on which backend can be initialized.
 * @return vector of CUDAs.
 */
KNP_DECLSPEC std::vector<CUDA> list_cuda_processors();

}  // namespace knp::devices::gpu
