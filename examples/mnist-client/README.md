Classifying images of handwritten numbers from the MNIST database. 
==============


# Solution overview

Example implementation of a neural network that classifies images of handwritten numbers from the MNIST database. 


# Solution file descriptions

## graph_physics.h header file description

The `graph_physics.h` header file contains definitions of the following interfaces:

* `PhysicsPoint` structure that defines a graph point and its physical attributes, such as index, position, and velocity.
* `VisualGraph` class that defines interface to graph visualization and the following functions:
    * `VisualGraph` constructor.
    * Iteration functions.
    * `scale_graph()` function that fits a graph to an image size.
    * `get_positions()` function that returns positions of graph points.
    * `get_force()` function that calculates a force from one graph point to another.


## graph_physics.cpp program file description

The `graph_physics.cpp` program file contains implementations of functions defined in the `graph_physics.h` header file.


## inference.h header file description 

The `inference.h` header file contains definitions for the following:

* `InferenceResult` structure that defines an interface used to store inference results for one population.
* `do_inference()` function that runs model inference for the specified number of steps on the specified backend using the specified data file.  


## inference.cpp program file description

The `inference.cpp` program file contains implementations of the following functions:

* `make_input_image_generator()` function that creates a spike message generator from an array of boolean frames.
* `find_projections()` function that finds a specific projection.
* `make_observer_function()` function that prints spike indexes in terminal. 
* `do_inference()` function defined in the `inference.h` header file.


## process_data.h header file description

The `process_data.h` header file contains definition for the `read_spikes_from_grayscale_file()` function. The function splits a grayscale image into a specified number of frames, converts each image frame into a boolean vector. If element value is `true`, it is considered to contain a spike.  


## process_data.cpp program file description

The `process_data.cpp` program file contains implementations of the following functions:

* `image_to_spikes()` function that transforms image data to spikes.
* `read_images_from_file()` function that reads images from a binary data file.
* `read_spikes_from_grayscale_file()` function defined in the `process_data.h` header file.


## main.cpp program file description

The `main.cpp` program file contains implementation of image classification of handwritten numbers from the MNIST database.



# Implementation of image classification of handwritten numbers from the MNIST database

`main.cpp` contains implementation of neural network that classifies images of handwritten numbers from the MNIST database.

_Image classification of handwritten numbers from the MNIST database consists of the following steps_:

1.  Header files required for the neural network execution are included using the `#include` directive.
2.  The `main()` function is implemented that loads a network and runs it or shows a subgraph of it depending on the specified task type.

    The `main()` function implements the following:
    
    1.  Creates an output ID.
    2.  Defines option values for task types, paths to network, a file storing data, and backend.
    3.  Creates an object that stores the defined options in the variable map. 
    4.  Defines the `main()` function behavior if the user calls the help option for command line switches information.
    5.  Defines the `main()` function behavior if a path to network is not provided.
    6.  Defines the `main()` function behavior if it is called to show a network subgraph. If the function is called to show a network subgraph, the `main()` function does the following:
        1.  Defines a path to network using a variable map.
        2.  Creates a network object and loads it from the defined path using the `sonata` namespace functions.
        3.  Build a visual graph using the `build_graph()` function.
        4.  Prints a textual description of graph connections using the `print_network_description()` function. 
        5.  Writes the graph to a DOT file using `write_dot()` function.
        6. Renders the pgraph to PNG using the `render_graph()` function.
    7. Defines the `main()` function behavior if it is called to run inference (classify images). If the function is called to run inference, the `main()` function does the following:
        1.  Defines a path to backend.
        2.  Runs model inference using the `do_inference()` function.


# Build

The CMake build system from the Kaspersky Neuromorphic Platform is used in a solution.

`CMakeLists.txt` contains CMake commands for building a solution.

If you install the `knp-examples` package, the example binary file is located in the `/usr/bin` directory. To execute the example binary file, run the following command:

`$ mnist-client`

You can also build the example by using CMake. The example binary file will be located in the `build/bin` directory. To execute the created binary file, run the following commands:

`$ build/bin/mnist-client`


# Information about third-party code

Information about third-party code is provided in the `NOTICE.txt` file located in the platform repository.


# Trademark notices

Registered trademarks and service marks are the property of their respective owners.\
Apache is either a registered trademark or a trademark of the Apache Software Foundation.

© 2026 AO Kaspersky Lab