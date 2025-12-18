Execution of a neural network automatically uploaded to the backend.
==============


# Solution overview

Example implementation of a neural network containing a population with one BLIFAT neuron, an input projection and a loop projection with one delta synapse each. The neural network is executed for 20 steps.


# Implementation of a neural network automatic upload and execution

`main.cpp` contains implementation of a neural network automatic upload and execution.

_Implementation of a neural network automatic upload to the backend and execution consists of the following steps_:

1.  Header files required for the neural network execution are included using the `#include` directive. Aliases are defined by means of the `using` statement.
2.  The `input_projection_gen` function is implemented that generates synapses for the projection connected to an input channel.
3.  The `synapse_generator` function is implemented that generates synapses for the projection, which loops the population output on itself.
4.  The `neuron_generator` function is implemented that generates neurons.
5.  The `main` function is implemented that creates the objects required for the neural network execution.

    The `main` function implements the following:
    
    1.  Creates a population object using `neuron_generator`.
    2.  Creates a projection object that loops the output of the population to itself is constructed using the associated population ID and `synapse_generator`.
    3.  Creates an input projection object using `synapse_generator` and associated population ID.
    4.  Creates a neural network object and adds the created projections and population to the neural network object.
    5.  Defines the IDs of the input and output channels.
    6.  Creates a model object using the neural network object, channel IDs, population IDs, and input projection IDs.
    7.  Creates a spike generation functor.
    8.  Specifies the path to the backend instance that executes the neural network.
    9.  Creates a model executor object using the model object, the path to the backend, the input channel ID, and the spike generation functor. 
    10. Runs the model execution.
6.  A vector is created that contains the indices of the steps at which spikes are received by the output channel.


# Build

The CMake build system from the Kaspersky Neuromorphic Platform is used in a solution.

`CMakeLists.txt` contains CMake commands for building a solution.

If you install the `knp-examples` package, the example binary file is located in the `/usr/bin` directory. To execute the example binary file, run the following command:

`$ simple-network`

You can also build the example by using CMake. The example binary file will be located in the `build/bin` directory. To execute the created binary file, run the following commands:

`$ cd /build/bin/simple-network`


# Information about third-party code

Information about third-party code is provided in the `NOTICE.txt` file located in the platform repository.


# Trademark notices

Registered trademarks and service marks are the property of their respective owners.

© 2024 AO Kaspersky Lab