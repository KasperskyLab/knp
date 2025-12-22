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

You can use the C++ and Python languages to accomplish these tasks. The platform supports CPUs, GPUs with CUDA support, as well as the AltAI-2 neuromorphic processing unit designed for power-efficient execution of neural networks on various classes of smart devices.

For information on the platform concepts and architecture, installation instructions and platform use cases, see <a href="https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=index">Kaspersky Neuromorphic Platform Help</a>.

Only versions of repository with release tags have release quality. If you use the source code from master branch or a build compiled from master branch, you may get code and build with errors and vulnerabilities.

## Hardware and software requirements

For Kaspersky Neuromorphic Platform operation, the computer must meet the following minimum requirements.

Minimum hardware requirements:

- CPU: Intel Core i5 or higher compatible processor
- GPU with CUDA support: NVIDIA GPU using Pascal or higher
- Neuromorphic processing unit (if needed): AltAI-2
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

For information on the required software, see <a href="https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=232788">Kaspersky Neuromorphic Platform Help</a>.

## Installation

You can install Kaspersky Neuromorphic Platform in one of the following ways:

- <a href="https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=273773">Install deb packages</a>
- <a href="https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=273774">Install Python development packages</a>
- <a href="https://click.kaspersky.com/?hl=en-US&version=2.0&pid=KNP&link=online_help&helpid=283082">Build a platform project</a>

## Trademark notices

Registered trademarks and service marks are the property of their respective owners.

AMD and AMD64 are trademarks or registered trademarks of Advanced Micro Devices, Inc.\
Apache is either a registered trademark or a trademark of the Apache Software Foundation.\
Apple, Leopard, Mac, Mac OS, macOS, OS X, Objective-C, Safari, TrueType and Xcode are trademarks of Apple Inc.\
Arm is a registered trademark of Arm Limited (or its subsidiaries) in the US and/or elsewhere.\
Borland is a trademark or registered trademark of Borland Software Corporation.\
Ubuntu and LTS are registered trademarks of Canonical Ltd.\
Docker and the Docker logo are trademarks or registered trademarks of Docker, Inc. in the United States and/or other countries. Docker, Inc. and other parties may also have trademark rights in other terms used herein.\
FreeBSD is a registered trademark of The FreeBSD Foundation.\
GITHUB is a trademark of GitHub, Inc., registered in the United States and other countries.\
GITLAB is a trademark of GitLab Inc. in the United States and other countries and regions.\
Google, Chrome and Chromium are trademarks of Google LLC.\
TensorFlow and any associated designations are trademarks of Google LLC.\
Intel, Core and Pentium are trademarks of Intel Corporation in the U.S. and/or other countries.\
Linux is the registered trademark of Linus Torvalds in the U.S. and other countries.\
Microsoft, Consolas, Internet Explorer, OpenType, Segoe, Visual Studio, Win32 and Windows are trademarks of the Microsoft group of companies.\
Firefox is a trademark of the Mozilla Foundation in the U.S. and other countries.\
NVIDIA is a registered trademark of NVIDIA Corporation.\
Java and JavaScript are registered trademarks of Oracle and/or its affiliates.\
Python is a trademark or registered trademark of the Python Software Foundation.\
Debian is a registered trademark of Software in the Public Interest, Inc.\
QT is a trademark or registered trademark of The Qt Company Ltd.\
UNIX is a registered trademark in the United States and other countries, licensed exclusively through X/Open Company Limited.

## Contribution

This is an open source project. If you are interested in making a code contribution, please see `CONTRIBUTING.md` for more information.

## License

Copyright © 2025 AO Kaspersky Lab\
Licensed under the Apache 2.0 License. See the `LICENSE.txt` file in the root directory for details.
