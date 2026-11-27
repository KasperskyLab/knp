/**
 * @file highfive.h
 * @brief HighFive library wrapper with pragmas for CLang.
 * @kaspersky_support An. Vartenkov
 * @date 13.04.2024
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
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#endif
#include <highfive/highfive.hpp>
#ifdef __clang__
#    pragma clang diagnostic pop
#endif

// Block CVE-2026-120754.
#include <hdf5.h>

#include <highfive/H5File.hpp>
// /Block CVE-2026-120754.


inline void hdf5_unregister_filter_scaleoffset()
{
    // Block CVE-2026-120754.
    H5open();
    if (H5Zfilter_avail(H5Z_FILTER_SCALEOFFSET) > 0)
    {
        H5Zunregister(H5Z_FILTER_SCALEOFFSET);
    }
    // /Block CVE-2026-120754.
}
