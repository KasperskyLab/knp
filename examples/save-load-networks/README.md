Saving and loading a simple neural network. 
==============


# Solution overview

Example implementation of creating a simple neural network and using `sonata` namespace functions to save and load it.


# Implementation of saving and loading a neural network

`main.cpp` contains implementation of saving and loading a neural network.

_Implementation of saving and loading a simple neural network consists of the following steps_:

1.  Header files required for the neural network execution are included using the `#include` directive. Aliases are defined by means of the `using` statement.
2.  The `input_projection_gen` function is implemented that generates synapses for the projection connected to an input channel.
3.  The `synapse_generator` function is implemented that generates synapses for the projection, which loops the population output on itself.
4.  The `neuron_generator` function is implemented that generates neurons.
5.  The `make_simple_network` function is implemented that returns a neural network.

    The `make_simple_network` function implements the following:

    1.  Creates a population object using `neuron_generator`.
    2.  Creates a projection object that loops the output of the population to itself is constructed using the associated population ID and `synapse_generator`.
    3.  Creates an input projection object using `synapse_generator` and associated population ID.
    4.  Creates a neural network object and adds the created projections and population to the neural network object.
6.  The `main` function is implemented that calls `sonata` namespace functions to save and load a network.

    The `main` function implements the following:

    1.  Creates a neural network using the `make_simple_network` function.
    2.  Saves a network using the `save_network` function.
    3.  Loads a network using the `load_network` function.


# Build

The CMake build system from the Kaspersky Neuromorphic Platform is used in a solution.

`CMakeLists.txt` contains CMake commands for building a solution.

If you install the `knp-examples` package, the example binary file is located in the `/usr/bin` directory. To execute the example binary file, run the following command:

`$ save-load-networks`

You can also build the example by using CMake. The example binary file will be located in the `build/bin` directory. To execute the created binary file, run the following commands:

`/build/bin/save-load-networks`


# Information about third-party code

Information about third-party code is provided in the `NOTICE.txt` file located in the platform repository.


# Trademark notices

Registered trademarks and service marks are the property of their respective owners.

© 2024 AO Kaspersky Lab