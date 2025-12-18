# Kaspersky Neuromorphic Platform

Platform for the spiking neural network execution.

<div align="center">
  <img src="packaging/neuromorphic-platform.svg" alt="Logo">
</div>

The Kaspersky Neuromorphic Platform ("KNP" or "platform") is a software platform for developing, training and executing spiking neural networks on a variety of computers.

You can use Kaspersky Neuromorphic Platform to do the following:

- Create spiking neural networks (SNNs) and train these on various types of input data, such as telemetry, events, images, 3D data, audio, and tactile data.
- Convert artificial neural networks (ANNs) into spiking networks and train these.
- Conduct applied research in the field of input data classification and other application domains of neural networks.
- Develop new neural network topologies, for example, impulse analogs of convolutional neural networks that involve convolution in space and time.
- Develop new models of synaptic plasticity.
- Implement new neuron models.
- Implement application solutions based on neuromorphic spiking neural networks in the field of robotic manipulators, Internet of Things, unmanned systems, human-machine interaction, wearable devices, and optimization planning.
- Implement application solutions on devices with low power consumption using neuromorphic processors.

You can use the C++ and Python languages to accomplish these tasks. The platform supports CPUs as well as the AltAI-1 neuromorphic processor unit designed for energy-efficient execution of neural networks in various types of intelligent devices.

For information on the platform concepts and architecture, installation instructions and platform use cases, see <a href="https://click.kaspersky.com/?hl=en-US&version=1.0&pid=KNP&link=online_help">Kaspersky Neuromorphic Platform Help</a>.

Only versions of repository with release tags have release quality. If you use the source code from master branch or a build compiled from master branch, you may get code and build with errors and vulnerabilities.

## Hardware and software requirements

For Kaspersky Neuromorphic Platform operation, the computer must meet the following minimum requirements.

Minimum hardware requirements:

- CPU: Intel Core i5 or higher compatible processor
- Neuromorphic processing unit (if needed): AltAI-1
- 8 GB of RAM
- Available hard drive space:
  - 1 GB for installing Kaspersky Neuromorphic Platform.
  - 10 GB for building the platform or an application solution build.

Supported operating systems:

- Debian GNU/Linux 12.5 or later
- Ubuntu 22.04 LTS or later
- Windows 7
- Windows 10

You can use the device running any other operating system from the Linux family, if the operating system distribution kit contains the Boost library version 1.81 or later.

To work with the platform, the following software must be installed on the device:

- Visual Studio 2022

  This software must be installed for building the platform or application solution in Windows.

- Boost 1.81 or later

  When working in Windows, it is recommended to install the library <a href="https://archives.boost.io/release/1.81.0/binaries/">precompiled</a> for Visual Studio 2022 (compiler version: 14.3).

- CMake 3.25 or later

- Before installing platform component deb packages for an `AltAI ANN2SNN` backend, make sure that the device has the following software:

  - TensorFlow 2.4.1 or later

  - Keras 2.3.1 or later

  - NumPy 2.4.1 or later

    The NumPy library version must match the TensorFlow library version.

- Before installing a whl or deb package containing a Python framework for an `AltAI SNN` backend, make sure that the device has Python 3.10 or later.

- Before installing a whl package containing a Python framework for an `AltAI ANN2SNN` backend, make sure that the device has the following software:

  - NumPy 1.24.3
  - TensorFlow 2.13.1
  - Loguru 0.7.2
  - PyYAML 6.0.1
  - NetworkX 3.1
  - Matplotlib 3.6.3
  - tqdm 4.66.5

- Before installing a deb package containing a Python framework for an `AltAI ANN2SNN` backend, make sure that the device has the following software:

  - python3-pip
  - python3-numpy
  - python3-yaml
  - python3-networkx
  - python3-matplotlib
  - python3-loguru
  - python3-tqdm
  - TensorFlow 2.13.1

## Trademark notices

Registered trademarks and service marks are the property of their respective owners.

Apache is either a registered trademark or a trademark of the Apache Software Foundation.\
Ubuntu and LTS are registered trademarks of Canonical Ltd.\
Docker and the Docker logo are trademarks or registered trademarks of Docker, Inc. in the United States and/or other countries. Docker, Inc. and other parties may also have trademark rights in other terms used herein.\
TensorFlow and any associated designations are trademarks of Google LLC.\
Intel and Core are trademarks of Intel Corporation in the U.S. and/or other countries.\
Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.\
Microsoft, Visual Studio and Windows are trademarks of the Microsoft group of companies.\
JavaScript is a registered trademark of Oracle and/or its affiliates.\
Python is a trademark or registered trademark of the Python Software Foundation.\
Debian is a registered trademark of Software in the Public Interest, Inc.

## Contribution

This is an open source project. If you are interested in making a code contribution, please see `CONTRIBUTING.md` for more information.

## License

Copyright © 2025 AO Kaspersky Lab
Licensed under the Apache 2.0 License. See the `LICENSE.txt` file in the root directory for details.
