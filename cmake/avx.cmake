#[[file avx.cmake
kaspersky_support Postnikov D.
date 16.01.2026
license Apache 2.0
copyright © 2025 AO Kaspersky Lab
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
#]]

function(enable_avx target)
  if(KNP_ENABLE_AVX)
    message(STATUS "AVX enabled for ${target}")
    target_compile_options(
      ${target}
      PRIVATE -march=native
              -mtune=native
              -mno-vzeroupper
              -Ofast
              -funroll-loops
              -fomit-frame-pointer
              -finline)
  else()
    message(WARNING "AVX is not enabled for ${target}, because AVX is disabled")
  endif()
endfunction()
