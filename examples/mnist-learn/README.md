Training neural network based on image data and labels and classifying images from the MNIST dataset. 
==============


# Solution overview

Example implementation of a neural network that trains on image data and labels and classifies images from the MNIST dataset. 


# Solution file descriptions

## construct_network.h header file description

The `construct_network.h` header file contains definitions of the following interfaces:

* `AnnotatedNetwork` structure that represents a neural network and stores network additional information such as IDs of output populations, IDs of projections from the raster input, IDs of populations from class labels, IDs of projections and populations used for inference, IDs of projections and populations used for Winner Takes All (WTA) processing, and names of populations. 
* `create_example_network` function that creates an instance of `AnnotatedNetwork` structure. In particular, the function creates a network with multiple populations and projections, and annotates it with network information. 


## construct_network.cpp program file description

The `construct_network.cpp` program file contains implementations of the following:

* `PopulationData` structure that stores population size and neuron parameters. 
* `PopIndexes` enum that is used to identify and index different populations in the neural network.
* `resource_from_weight` function that calculates the synaptic resource value based on minimum and maximum weight values.
* `add_subnetwork_populations` function that adds populations to the network. The function is called by the `create_example_network` function.
* `create_example_network` function defined in the `construct_network.h` header file.


## inference.h header file description 

The `inference.h` header file contains definition for the `run_mnist_inference` function used to run model inference on the MNIST dataset.  


## inference.cpp program file description

The `inference.cpp` program file contains implementation of the `run_mnist_inference` function defined in the `inference.h` header file. The function loads the specified backend, created a model from trained network `AnnotatedNetwork`, creates a model executor, adds input and output channels to the model executor, and sets up logging for inference data. The function then starts model executor to run inference and returns a vector of spikes generated during network inference.


## shared_network.h header file description

The `shared_network.h` header file contains definitions of the neural network architecture and hyperparameters. 


## time_string.h header file description

The `time_string.h` header file contains definition for the `get_time_string` function. The function is used to get a string representation of the current time.  


## time_string.cpp program file description

The `time_string.cpp` program file contains implementation of the `get_time_string` function defined in the `time_string.cpp` header file.


## visualize_network.h header file description

The `train.h` header file contains definitions for the following:

* `num_subnetworks` constant that defines the number of subnetworks to use for training.
* `train_mnist_network` function that is used to train a neural network on the MNIST dataset.


## visualize_network.cpp program file description 

The `train.cpp` program file contains implementations of the following:

* `aggregated_spikes_logging_period` constant that defines the logging period for aggregated spikes.
* `projection_weights_logging_period` constant that defines the logging period for aggregated weights. 
* `wta_winners_amount` constant that defines the number of winners to select for WTA processing. 
* `build_channel_map_train` function that creates a channel map for training.
* `get_network_for_inference` function that returns a network for inference.
* `train_mnist_network` function defined in the `train.h` header file. The function creates the network `AnnotatedNetwork` using the `create_example_network` function defined in the `construct_network.h` header file and saves it to a file using the `sonata` namespace function. The function then creates a model and model executor, sets up input channels for the model executor using the `build_channel_map_train` function, and adds logging for the training data. Finally, the function starts the model executor to train network on the MNIST dataset and returns the trained network `AnnotatedNetwork`.


## MNIST.zip file 

The `MNIST.zip` file contains image data and corresponding labels for the MNIST dataset.


## main.cpp program file description

The `main.cpp` program file contains implementation of neural network training on image data and labels and classification of images from the MNIST database.



# Implementation of neural network trainng on image data and labels and classification of images from the MNIST database

`main.cpp` contains implementation of neural network that trains on image data and labels from the MNIST database and classifies images.

_Neural network trainng on image data and labels and classification of images from the MNIST database consists of the following steps_:

1. Header files required for the neural network execution are included using the `#include` directive.
2. Constants required for the input data processing, such as the number of steps to process an image and the number of classes in the dataset, are defined.
3. The `main` function is implemented that processes the images and their labels, splits the dataset into training and test sets, trains the neural network on the training set and runs inference on the test set. 

    The `main` function implements the following:
    
    1.  Checks if the paths to images and labels files and the path to log folder were provided as arguments for the function.
    2.  Reads the paths to images and labels files and the path to log folder.
    3.  Defines the `std::filesystem::path` object to store path to the backend.
    4.  Defines the `knp::framework::data_processing::classification::images::Dataset` object to store the MNIST dataset.
    5.  Defines the `std::ifstream` objects to read images and labels from the specified paths.
    6.  Processes images and labels files from the `std::ifstream` objects using the `process_labels_and_images` function.
    7.  Splits the MNIST dataset into training and test sets using the `split` function.
    8.  Trains a neural network model using the `train_mnist_network` function and stores the trained model in an `AnnotatedNetwork` object.
    9.  Runs inference on the test set using the `run_mnist_inference` function, and stores output spikes in a vector.
    10. Evaluates the results the `process_inference_results` function.
    11. Writes evaluation results to console and log file using the `write_inference_results_to_stream_as_csv` function.


# Prerequisites

Before running the example, unpack the `MNIST.zip` archive.

# Build

The CMake build system from the Kaspersky Neuromorphic Platform is used in a solution.

`CMakeLists.txt` contains CMake commands for building a solution.

If you install the `knp-examples` package, the example binary file is located in the `/usr/bin` directory. To execute the example binary file, run the following command:

`$ mnist_learn_example MNIST.bin MNIST.target <optional directory to store log files>`

You can also build the example by using CMake. The example binary file will be located in the `build/bin` directory. To execute the created binary file, run the following commands:

`$ build/bin/mnist_learn_example MNIST.bin MNIST.target <optional directory to store log files>`


# Information about third-party code

Information about third-party code is provided in the `NOTICE.txt` file located in the platform repository.


# Trademark notices

Registered trademarks and service marks are the property of their respective owners.

© 2025 AO Kaspersky Lab