#!/usr/bin/env python
"""
@file cccc.py
@brief Wrapper script for CCCC.

@license Apache 2.0 License.
@copyright © 2024-2025 AO Kaspersky Lab
@date 28.10.2024.

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

from json import load
import sys
from pathlib import Path
from xml.etree import ElementTree

from tools_lib.static_analyzer_cmd import StaticAnalyzerCmd  # pylint: disable = E0401


class CcccCmd(StaticAnalyzerCmd):  # type: ignore
    """Class for the cccc command."""

    command = 'cccc'

    def __init__(self, args: list[str]):
        super().__init__(self.command, '', args)
        self.check_installed()
        self.parse_args(args)
        self.artifacts_dir = Path().cwd() / '.cccc'
        self.config_name = 'cccc.xml'
        self.limits_file = Path('.cccc.json')

        for a in self.args:
            opt_name, opt_val = a.split('=')
            if opt_name.startswith('--outdir'):
                self.artifacts_dir = Path.cwd() / opt_val
            elif opt_name.startswith('--xml_outfile'):
                self.config_name = opt_val
            elif opt_name.startswith('--limits_file'):
                self.limits_file = Path(opt_val)
                self.args.remove(a)
        self.config_file = self.artifacts_dir / self.config_name

    def _make_command(self, file: Path) -> list[str]:
        self.args: list[str]
        file_s = str(file)

        if file_s.lower().endswith(('.cu', '.cuh')):
            # CUDA fix.
            return self.args + ['--lang=c++', file_s]

        return self.args + [file_s]

    def _process_param(self, file: Path, module: ElementTree.Element, limits: dict[str, int]) -> None:
        module_data = {}
        module_name = ''

        for param in module:
            if 'name' == param.tag:
                module_name = param.text
            elif 'lines_of_code' == param.tag:
                module_data['locf_max'] = int(param.attrib['value'])
            elif 'McCabes_cyclomatic_complexity' == param.tag:
                module_data['mvg_function_max'] = int(param.attrib['value'])
            elif 'McCabes_cyclomatic_complexity_per_module' == param.tag:
                module_data['mvg_file_max'] = int(param.attrib['value'])
            elif 'rejected_lines_of_code' == param.tag:
                module_data['rejected_lines_of_code'] = int(param.attrib['value'])

        for i, v in limits.items():
            if module_data.get(i, 0) > v:
                self.raise_error(
                    f'{i} in "{file}" [{module_name}] == {module_data[i]}', f'This exceeds the limit of {v}.'
                )
                sys.exit(1)

    def run(self) -> None:
        """Run cccc."""

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        with open(self.limits_file, encoding='utf8') as f:
            limits = load(f)

        for file in self.files:
            self.run_command(self._make_command(file))
            self.exit_on_error()

            try:
                tree = ElementTree.parse(self.config_file)
            except ElementTree.ParseError as e:
                print(f'WARNING: CCCC BUG: {e}.')
                return

            root = tree.getroot()

            for module in root.findall('./procedural_summary/'):
                self._process_param(file, module, limits)


def main(argv: list[str]) -> None:
    cmd = CcccCmd(argv)
    cmd.run()


if __name__ == '__main__':
    main(sys.argv)
