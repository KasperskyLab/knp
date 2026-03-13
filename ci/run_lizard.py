#!/usr/bin/env python3
"""
@file run_lizard.py
@brief Run lizard static analyzer.

@kaspersky_support David P.
@license Apache 2.0 License.
@copyright © 2026 AO Kaspersky Lab
@date 13.03.2026

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import subprocess


def run_lizard(path: str, nloc: int, complexity: int, token_count: int, arguments: int) -> None:
    print(f"Running lizard in {path}")
    command = [
        "lizard",
        path,
        "-T",
        f"nloc={nloc}",
        "-T",
        f"cyclomatic_complexity={complexity}",
        "-T",
        f"token_count={token_count}",
        "-T",
        f"parameter_count={arguments}",
        "-w",
    ]
    subprocess.run(command, check=False)
    print()


run_lizard("knp", 50, 10, 300, 4)
run_lizard("examples", 100, 15, 500, 6)
