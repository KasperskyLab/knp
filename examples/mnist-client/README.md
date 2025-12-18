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
    * `scale_graph` function that fits a graph to an image size.
    * `get_positions` function that returns positions of graph points.
    * `get_force` function that calculates a force from one graph point to another.


## graph_physics.cpp program file description

The `graph_physics.cpp` program file contains implementations of functions defined in the `graph_physics.h` header file.


## inference.h header file description 

The `inference.h` header file contains definitions for the following:

* `InferenceResult` structure that defines an interface used to store inference results for one population.
* `do_inference` function that runs model inference for the specified number of steps on the specified backend using the specified data file.  


## inference.cpp program file description

The `inference.cpp` program file contains implementations of the following functions:

* `make_input_image_generator` function that creates a spike message generator from an array of boolean frames.
* `find_projections` function that finds a specific projection.
* `make_observer_function` function that prints spike indexes in terminal. 
* `do_inference` function defined in the `inference.h` header file.


## process_data.h header file description

The `process_data.h` header file contains definition for the `read_spikes_from_grayscale_file` function. The function splits a grayscale image into a specified number of frames, converts each image frame into a boolean vector. If element value is `true`, it is considered to contain a spike.  


## process_data.cpp program file description

The `process_data.cpp` program file contains implementations of the following functions:

* `draw_data` function that visualizes data in the form of graph for the specified vector of images.
* `image_to_spikes` function that transforms image data to spikes.
* `read_images_from_file` function that reads images from a binary data file.
* `read_spikes_from_grayscale_file` function defined in the `process_data.h` header file.


## visualize_network.h header file description

The `visualize_network.h` header file contains definitions for the following:

* `NetworkGraph` structure that defines network graph attributes and constructor. The structure includes definitions for the following: 
    * `Node` structure that defines graph node attributes.
    * `Edge` structure that defines graph edge attributes.
* `print_network_description` function that prints description of connections between graph nodes and edges.
* `print_modified_network_description` function that prints description of connections between graph nodes and edges in the machine-readable format.
* `divide_graph_by_connectivity` function that divides network graph into subgraphs.
* `position_network` function that positions a subgraph for drawing.
* `position_network_test` function that draws a subgraph throughout the positioning process.


## visualize_network.cpp program file description 

The `visualize_network.cpp` program file contains implementations of the following:

* `DrawingParameters` structure that defines attributes for network graph drawing.
* `get_name` function that gets an entity name or ID for any network object.
* `NetworkGraph` constructor defined in the `visualize_network.h` file.
* `build_adjacency_list` function that transforms a network graph into an adjacency list. 
* `draw_simple_arrow_line` function that draws an arrow from one specified graph point to another graph point.
* `draw_edges` function that draws edges between two specified graph nodes.
* `draw_annotated_subgraph` function that draws a network graph with inscriptions for node names or IDs.
* `draw_subgraph` function that draws a connected subgraph.
* `make_reverse_list` function that creates a reverse adjacency list where each node has a list of nodes to which it is adjacent.
* `find_connected_set` function that finds an independent subgraph inside a network graph.
* `divide_graph_by_connectivity` function defined in the `visualize_network.h` file.
* `print_connected_subset` function that prints the description of a network subgraph.
* `print_network_description` function defined in the `visualize_network.h` file.
* `position_network_test` function defined in the `visualize_network.h` file.
* `position_network` function defined in the `visualize_network.h` file.


## main.cpp program file description

The `main.cpp` program file contains implementation of image classification of handwritten numbers from the MNIST database.



# Implementation of image classification of handwritten numbers from the MNIST database

`main.cpp` contains implementation of neural network that classifies images of handwritten numbers from the MNIST database.

_Image classification of handwritten numbers from the MNIST database consists of the following steps_:

1.  Header files required for the neural network execution are included using the `#include` directive.
2.  The `main` function is implemented that loads a network and runs it or shows a subgraph of it depending on the specified task type.

    The `main` function implements the following:
    
    1.  Creates an output ID.
    2.  Defines `std::string` objects for task types, path to network and path to file storing data. 
    3.  Defines option values for task types, path to network and path to file storing data.
    4.  Creates an object that stores the defined options in the variable map. 
    5.  Defines the `main` function behavior if a path to network is not provided.
    6.  Defines the `main` function behavior if it is called to show a network subgraph. If the function is called to show a network subgraph, the `main` function does the following:
        1.  Defines a path to network using a variable map.
        2.  Creates a network object and loads it from the defined path using the `sonata` namespace functions.
        3.  Creates a network graph using the `NetworkGraph` structure.
        4.  Prints descriptions of graph connections using the `print_network_description` function. 
        5.  Draws a subgraph using the `position_network_test` function.
    7. Defines the `main` function behavior if it is called to run inference (classify images). If the function is called to run inference, the `main` function does the following:
        1.  Defines a path to backend.
        2.  Runs model inference using the `do_inference` function.


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

Registered trademarks and service marks are the property of their respective owners.

© 2024 AO Kaspersky Lab