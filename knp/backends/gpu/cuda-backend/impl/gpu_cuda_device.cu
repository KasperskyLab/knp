/**
 * @file gpu_cuda_device.cu
 * @brief CUDA device class implementation.
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

#include <knp/devices/gpu_cuda.h>

#include <spdlog/spdlog.h>

#include <nvml.h>

#include <exception>


namespace knp::devices::gpu
{

CUDA::CUDA(uint32_t gpu_num) : gpu_num_(gpu_num)
{
    if (const auto e_code = cudaGetDeviceProperties(&properties_, gpu_num_); e_code != cudaSuccess)
    {
        const auto err_msg = std::string("CUDA error during get device properties") + cudaGetErrorString(e_code);
        SPDLOG_ERROR("{} [{}]", err_msg.c_str(), e_code);
        throw std::runtime_error(err_msg);
    }

    static_assert(sizeof(boost::uuids::uuid) == sizeof(cudaUUID_t));
    std::copy_n(reinterpret_cast<const char*>(&properties_.uuid),
                Device::base_.uid_.tag.size(), Device::base_.uid_.tag.begin());
}


CUDA::CUDA(CUDA&& other)
    : gpu_num_{std::move(other.gpu_num_)},
      properties_{std::move(other.properties_)}
{
}


CUDA::~CUDA()
{
}


CUDA& CUDA::operator=(CUDA&& other) noexcept
{
    std::swap(gpu_num_, other.gpu_num_);
    std::swap(properties_, other.properties_);

    return *this;
}


knp::core::DeviceType CUDA::get_type() const
{
    return knp::core::DeviceType::GPU;
}


const std::string CUDA::get_name() const
{
    return properties_.name;
}


uint32_t CUDA::get_socket_number() const
{
    return gpu_num_;
}


float CUDA::get_power() const
{
    try
    {
        if (auto e_code = nvmlInit_v2(); e_code != NVML_SUCCESS)
        {
            throw std::runtime_error(std::string("NVML initialization failed: ") + std::to_string(e_code));
        }

        nvmlDevice_t device_handle;

        if (auto e_code = nvmlDeviceGetHandleByIndex_v2(gpu_num_, &device_handle); e_code != NVML_SUCCESS)
        {
            throw std::runtime_error(std::string("Device handle getting error: ") + std::to_string(e_code));
        }

        nvmlEnableState_t mode;
        if (auto e_code = nvmlDeviceGetPowerManagementMode(device_handle, &mode); e_code != NVML_SUCCESS)
        {
            throw std::runtime_error(std::string("Device power management mode getting error: ") +
                                     std::to_string(e_code));
        }

        unsigned int power_usage;
        if (auto e_code = nvmlDeviceGetPowerUsage(device_handle, &power_usage); e_code != NVML_SUCCESS)
        {
            throw std::runtime_error(std::string("Device power usage getting error: ") + std::to_string(e_code));
        }

        if (auto e_code = nvmlShutdown(); e_code != NVML_SUCCESS)
        {
            SPDLOG_WARN("NVML shutdown failed: {}", e_code);
        }

        return power_usage;
    }
    catch(const std::runtime_error &e)
    {
        SPDLOG_ERROR("{}", e.what());
        if (auto e_code = nvmlShutdown(); e_code != NVML_SUCCESS)
        {
            SPDLOG_WARN("NVML shutdown failed: {}", e_code);
        }
        throw;
    }
    catch(...)
    {
        throw;
    }
}


unsigned int CUDA::get_warp_size() const
{
    return static_cast<unsigned int>(properties_.warpSize);
}


unsigned int CUDA::get_mp_count() const
{
    return static_cast<unsigned int>(properties_.multiProcessorCount);
}


unsigned int CUDA::get_threads_per_mp() const
{
    return static_cast<unsigned int>(properties_.maxThreadsPerMultiProcessor);
}


unsigned int CUDA::get_max_threads_count() const
{
    return static_cast<unsigned int>(properties_.maxThreadsPerBlock);
}


unsigned int CUDA::get_concurrent_kernels() const
{
    return static_cast<unsigned int>(properties_.concurrentKernels);
}


const std::array<int, 3> CUDA::get_max_threads_dim() const
{
    return {properties_.maxThreadsDim[0], properties_.maxThreadsDim[1], properties_.maxThreadsDim[2]};
}


const std::array<int, 3> CUDA::get_max_grid_size() const
{
    return {properties_.maxGridSize[0], properties_.maxGridSize[1], properties_.maxGridSize[2]};
}


unsigned int CUDA::get_global_memory_bytes() const
{
    return static_cast<unsigned int>(properties_.totalGlobalMem);
}


unsigned int CUDA::get_constant_memory_bytes() const
{
    return static_cast<unsigned int>(properties_.totalConstMem);
}


KNP_DECLSPEC std::vector<CUDA> list_cuda_processors()
{
    int device_count = 0;

    if (auto e_code = cudaGetDeviceCount(&device_count); e_code != cudaSuccess)
    {
        const auto err_msg = std::string("CUDA error during get device properties") + cudaGetErrorString(e_code);
        SPDLOG_ERROR("{} [{}]", err_msg.c_str(), e_code);
        throw std::runtime_error(err_msg);
    }

    if (0 == device_count) return {};

    std::vector<CUDA> result;
    result.reserve(device_count);

    for (int i = 0; i < device_count; ++i) result.push_back(CUDA(i));

    return result;
}

}  // namespace knp::devices::gpu
