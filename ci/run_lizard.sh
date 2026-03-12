#!/bin/sh -x

# @file run_lizard.sh
# @kaspersky_support David P.
# @license Apache 2.0
# @copyright © 2026 AO Kaspersky Lab
#date 12.03.2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


echo "Running lizard on knp."
lizard -C 10 -a 4 -T nloc=50 -T token_count=300 knp -w

echo "Running lizard on examples."
lizard -C 15 -a 6 -T nloc=100 -T token_count=500 examples -w
