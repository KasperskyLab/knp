Training neural network based on image data and labels and classifying images from the MNIST dataset. 
==============


# Solution overview

Example implementation of a neural network that trains on image data and labels and classifies images from the MNIST dataset. 

This example implements a complete pipeline for training and evaluating neural networks on the MNIST dataset. It features:

* `SynapticResourceSTDPAltAILIFNeuron` and `SynapticResourceSTDPBLIFATNeuron` neuron models
* Network construction and training utilities
* Dataset handling for MNIST images
* Evaluation and result analysis
* Network serialization and inference capabilities


# Solution file descriptions

## models directory

The directory contains the complete implementation of two distinct neural network neuron models (AltAI and BLIFAT) designed for MNIST digit recognition tasks. 


### altai directory

The directory contains the complete implementation of the AltAI neuron model for neural network training and inference.


#### construct_network.cpp program file description

The `construct_network.cpp` file implements the network construction logic specifically for the AltAI neuron model in the MNIST neural network example. This file is responsible for creating the complete network architecture with appropriate neuron parameters, population structures, and synaptic connections. The `construct_network.cpp` file implements the following:

* `NetworkPopulations` structure that stores all populations used in the neural network.
* `create_populations()` function that creates neuron parameter structures with specialized AltAI parameters. The function also configures different neuron properties for input and output populations. Input neurons have specific leak rates and threshold adjustments, whereas output neurons use default parameters for classification. The function creates five distinct population types:
    * Input population — processes rasterized image data with specialized neuron parameters.
    * Output population — produces classification results.
    * Gate population — controls competitive learning mechanisms for WTA (Winner-Take-All) operations.
    * Raster population — handles image channeling.
    * Target population — processes label information.
* `create_projections()` function that establishes all synaptic connections between network populations with appropriate weights, delays, and connection patterns. The function creates the following projections:
    * Raster to input projection. Raster population (image data) sends spikes based on pixel values to the input population, STDP plasticity adjusts synaptic weights based on spike timing.
    * Target to input (dopamine) projection. The correct labels act as reward or error signals that tell the network whether its predictions were correct or incorrect. When the network makes a correct prediction, the dopamine signal reinforces the successful pathways. When predictions are incorrect, the signal helps adjust the network toward better performance.
    * Target to input (excitatory) projection. The correct labels send excitatory signals to the appropriate input neurons to reinforce the correct pathway.
    * Target to gate. The label information is fed back to gate neurons to influence the network's processing behavior. The labels activate gate neurons that modulate the activity of input neurons. The gate neurons help implement the WTA mechanism by controlling which neurons become active. This creates a closed-loop learning system where the network receives both bottom-up sensory input and top-down label information.
    * Input to output projection. The connection defines the pathway through which the network transforms visual input patterns into discrete classification decisions. The strength and connectivity pattern of this projection determines how well the network can distinguish between different digit classes based on the input spike patterns.
    * Output to gate projection. This connection sends signals from the output population back to the gate population, creating a feedback loop that influences the competitive learning process.
    * Gate to input projection. The connection creates a modulatory feedback loop where gate neurons act as "control signals" that influence how input neurons respond based on information received from the output neurons.
* `construct_network()` function that orchestrates the complete network construction process:
    * Creates the complete network structure using `create_populations()` and `create_projections()` functions
    * Calculates WTA borders based on neuron column organization
    * Configures sender-receiver relationships for competitive learning
    * Integrates all populations and projections into the annotated network structure


#### hyperparameters.h header file description

The `hyperparameters.h` file defines the specialized hyperparameters for the AltAI neuron model implementation. These parameters are carefully tuned for the AltAI neural network architecture and include both standard neural parameters and model-specific scaling factors.


#### network_functions.h header file description

The `network_functions.h` file declares the specialized versions of generic network functions tailored for the AltAI neuron model:

* `construct_network()` function implemented in the `construct_network.cpp` file.
* `prepare_network_for_inference()` function implemented in the `prepare_network_for_inference.cpp` file.
* `make_training_labels_spikes_generator()` function implemented in the `spike_generators.cpp` file.


#### prepare_network_for_inference.cpp program file description

The `prepare_network_for_inference.cpp` file implements the specialized network preparation logic required for running AltAI neural networks inference. The `prepare_network_for_inference.cpp` file implements the following functions:

* `replace_wta_with_projections()` function that:
    * Replaces WTA mechanisms with direct projection connections. This is necessary because the AltAI model does not natively support WTA operations.
    * Copies existing projections and reconfigures them for inference compatibility.
    * Clears WTA data after replacement to prevent conflicts.
* `quantize_network()` function that: 
    * Scales all network weights and thresholds to the range [-255, 255].
    * Addresses AltAI model limitations requiring fixed-point arithmetic.
    * Normalizes weights based on maximum values in each projection.
    * Applies scaling factors to both synaptic weights and neuron thresholds.
    * Rounds values to maintain integer precision.
* `prepare_network_for_inference()` function that:
    * Restores network from backend data with inference-specific population and projection filtering.
    * Selects only populations and projections marked for inference use.
    * Applies both WTA replacement and network quantization.
    * Prepares the network for efficient inference execution.


#### spike_generators.cpp program file description

The `spike_generators.cpp` file implements the `make_training_labels_spikes_generator()` function that generates spike patterns from training labels at specific time intervals in the AltAI neuron model.

### blifat directory

The directory contains the complete implementation of the BLIFAT neuron model for neural network training and inference.


#### construct_network.cpp program file description

The `construct_network.cpp` file implements a BLIFAT-specific network construction that follows the same general architectural patterns and framework as the AltAI implementation, but with distinct neuron models, parameter sets, and architectural choices (such as multi-subnetwork design) tailored specifically for BLIFAT neuron behavior.


#### hyperparameters.h header file description

The `hyperparameters.h` file defines the specialized hyperparameters for the BLIFAT neuron model implementation. These parameters are carefully tuned for the BLIFAT spiking neural network architecture and include both standard neural parameters and model-specific configurations.


#### network_functions.h header file description

The `network_functions.h` file declares the specialized versions of generic network functions tailored for the BLIFAT neuron model:

* `construct_network()` function implemented in the `construct_network.cpp` file.
* `prepare_network_for_inference()` function implemented in the `prepare_network_for_inference.cpp` file.
* `make_training_labels_spikes_generator()` function implemented in the `spike_generators.cpp` file.


#### prepare_network_for_inference.cpp program file description

The `prepare_network_for_inference.cpp` file implements the specialized network preparation logic required for running BLIFAT spiking neural networks in inference mode. The file implements the `prepare_network_for_inference()` function that:

* Extracts only the necessary populations and projections from the training backend
* Uses the annotation data to select only components marked for inference use
* Preserves the original BLIFAT network structure while removing unnecessary components


#### spike_generators.cpp program file description

The `spike_generators.cpp` file implements the `make_training_labels_spikes_generator()` function that generates spikes for training label data in the BLIFAT neuron model.


### network_constructor.h header file description

The `network_constructor.h` file provides a comprehensive framework for building spiking neural networks with proper annotation and management of populations and projections. The file defines the following components:

* `PopulationRole` enum that defines population roles.
* `PopulationInfo` structure that stores metadata for each network population.
* `NetworkConstructor` class that provides the following methods:
    * `add_population()` method that creates populations with specified neuron parameters and assigns roles and manages inference retention.
    * `add_channeled_population()` method that creates channelized populations.
    * `add_projection()` method that creates synaptic connections between populations, handles training flags and WTA connectivity, and manages inference projection retention.


### network_functions.h header file description

The `network_functions.h` file provides the generic template declarations for neuron model-specific network functions:

* `construct_network()` function that provides the interface for network construction across all neuron models.
* `prepare_network_for_inference()` function that provides the interface for network preparation before inference.
* `make_training_labels_spikes_generator()` function that provides the interface for creating training label spike generators.


### resource_from_weight.h header file description

The `resource_from_weight.h` file provides the `resource_from_weight()` function. The function calculates synaptic resource values from weight parameters. This function is essential for resource-based plasticity mechanisms in neural networks that use synaptic resource management.


## annotated_network.h header file description

The `annotated_network.h` header file contains definition for the `AnnotatedNetwork` structure. This structure extends the basic network framework by adding semantic annotations that provide additional context and metadata about the network's structure and behavior. The annotation metadata enables the following:

* Selective inference. During inference operations, only specified populations and projections are maintained, optimizing memory usage and computational efficiency.
* WTA management. Proper handling of WTA mechanisms through tracked sender-receiver pairs and boundary definitions.
* Network analysis. Easy identification of key network components for debugging, visualization, and performance monitoring.
* Modular operations. Clear separation between network structure and semantic meaning, facilitating complex network management.
 

## dataset.cpp program file description

The `dataset.cpp` file contains the core implementation for processing and preparing the MNIST dataset for spiking neural network training and inference.

The main function `process_dataset()` performs the following operations:

* File validation. The function checks that both images and labels files exist before proceeding.
* Stream creation. The function opens binary streams for reading image data and text streams for labels.
* Dataset processing. The function converts raw MNIST data into spike representations using the configured converter.
* Data splitting. The function divides the processed dataset into training and inference portions based on model configuration.
* Statistics reporting. The function outputs processing results including step counts for training and inference phases.


## dataset.h header file description

The `dataset.h` header file provide interface to the `process_dataset()` function implemented in the `dataset.cpp` file. This function serves as the entry point for dataset preparation in this example. It abstracts away the complexity of file input and output, data conversion, and dataset splitting, providing a clean interface for the rest of the application to work with structured, ready-to-use dataset objects.


## evaluate_results.cpp program file description

The `evaluate_results.cpp` file contains the implementation of the `evaluate_results()` function. This function is used for processing and evaluating the results of neural network inference operations on the MNIST dataset. The `evaluate_results()` function performs the following operations:

* Result processing. The function uses `InferenceResultsProcessor` from the KNP framework to analyze inference spikes.
* Dataset integration. The function combines spike data with the original dataset for accurate evaluation.
* CSV output. The function writes formatted evaluation results to standard output in CSV format.

This function is typically called after completing inference operations to assess network performance. The output includes classification accuracy metrics and detailed performance statistics that help determine the effectiveness of the trained spiking neural network.


## evaluate_results.h header file description

The `evaluate_results.h` header file provides the interface to the `evaluate_results()` function implemented in the `evaluate_results.cpp` file. 


## global_config.h header file description

The `global_config.h` file defines essential constant parameters for the MNIST neural network implementation. These constants provide a centralized configuration point for the entire example. They define the fundamental characteristics of the neural network implementation, including:

* Input and output dimensions
* Simulation timing parameters
* Network topology settings
* Logging frequency controls


## inference.h header file description 

The `inference.h` file provides the interface for running inference operations on neural networks with support for different neuron models. The `inference.h` file defines the following functions:

* `infer_network()` — template function that executes inference on a given network:

    * Sets up the neural network model with appropriate input and output channels
    * Configures WTA mechanisms for competitive learning
    * Initializes logging infrastructure for spike monitoring
    * Executes the inference simulation with progress reporting
    * Returns recorded spike messages for evaluation

* `infer_model()` — template function that orchestrates the complete inference process:

    * Loads the appropriate backend for inference execution
    * Delegates to `infer_network()` for the actual inference computation
    * Handles backend management and resource allocation

Both functions are templated on `Neuron` type, allowing for flexible neuron model support including:

* AltAI neuron implementations
* BLIFAT neuron implementations


## model_desc.cpp program file description

The `model_desc.cpp` file implements the output streaming operator for the `ModelDescription` structure, providing a convenient way to display comprehensive model configuration information. The `operator<<` function formats and outputs all relevant model configuration parameters to any output stream, making it easy to inspect and debug model settings.


## model_desc.h header file description

The `model_desc.h` file defines the configuration structure and streaming interface for neural network models, providing a centralized way to manage model parameters and display configuration information. The `model_desc.h` file defines the following interfaces:

* `SupportedModelType` — enum that defines the two supported neuron models for the MNIST example.
* `ModelDescription` — structure that contains all configurable parameters that can be modified via command line or configuration files:
    * Model type selection
    * Training and inference dataset sizes
    * File paths for images, labels, and backend libraries
    * Logging and model saving paths
* `operator<<` — stream output operator that provides human-readable formatting of model configuration. The operator is defined in the `model_desc.cpp` file.


## parse_arguments.cpp program file description

The `parse_arguments.cpp` file implements the command-line argument parsing functionality for the MNIST neural network example. The comprehensive command-line interface management is provided via the `parse_arguments()` function. The implementation supports the following command-line options:

* `--model` or `-m` for model type selection
* `--train_iters` and `--inference_iters` for defining training and inference image counts
* `--images` and `--labels` for defining paths to dataset images and labels files
* `--training_backend` and `--inference_backend` for defining computational backends used for training and inference
* `--extensive_logs_path` and `--logging_level` for logging control
* `--model_path` for saving trained models
* `--help` or `-h` for displaying usage information


## parse_arguments.h header file description

The `parse_arguments.h` file provides the interface to the `parse_arguments()` function implemented in the `parse_arguments.cpp` file.


## plot_spikes_raster.py program file description

The `plot_spikes_raster.py` file is a Python script designed to visualize spike activity from spiking neural network inference runs. It generates raster plots that display the temporal and spatial patterns of neural activity, making it easier to analyze the behavior of spiking networks.

The script reads spike data from `spikes_inference_raw.csv` files and validates CSV structure and required columns. If required, you can filter by sender names for focused analysis:

`$python plot_spikes_raster.py spikes_inference_raw.csv --sender "<sender_name>"`

The scrypt then creates high-quality raster plots showing spike timing against neuron positions and automatically calculates appropriate figure dimensions based on data. The script produces a PNG image showing:

* X-axis: Time steps of spike occurrences
* Y-axis: Stacked neuron indices grouped by network components
* Black vertical lines representing individual spikes
* Labeled tracks for different network populations
* Grid lines for better readability


## save_network.h header file description

The `save_network.h` file provides functionality for persisting trained neural networks in SONATA format, enabling model persistence and reuse across different sessions.
The file defines the `save_network()` function that creates the target directory for model saving if it doesn't exist and saves the trained network structure to the specified path in SONATA format. The function integrates seamlessly with the existing `ModelDescription` and `AnnotatedNetwork` structures. This function is typically called after successful training completion to preserve the learned network weights and structure.

## training.h header file description

The `training.h` file provides the complete implementation for training spiking neural networks on the MNIST dataset, supporting both BLIFAT and AltAI neuron models through template-based generic programming. The `training.h` file defines the following functions:

* `build_channel_map_train()` — a function that:
    * Sets up input and output channels for training operations
    * Configures spike generators for image data and labels
    * Creates the channel mapping required by the framework
    * Supports both rasterized image inputs and class label projections
* `train_network()` — a function that:
    * Executes the complete training process for a given network
    * Manages model initialization and execution
    * Configures comprehensive logging for training monitoring
    * Handles WTA mechanisms for competitive learning
    * Provides real-time progress updates during training
* `train_model()` — a function that:
    * Orchestrates the complete training workflow
    * Loads the appropriate training backend
    * Delegates to `train_network()` for actual training
    * Prepares the network for inference after training completion


## MNIST.zip file 

The `MNIST.zip` file contains image data and corresponding labels for the MNIST dataset.


## main.cpp program file description

The `main.cpp` file serves as the entry point and orchestration hub for the complete MNIST neural network learning pipeline. This file coordinates all major components of the system, from command-line argument parsing to network training, inference, and evaluation.



# Implementation of neural network training on image data and labels and classification of images from the MNIST database

`main.cpp` contains implementation of neural network that trains on image data and labels from the MNIST database and classifies images.

_Neural network training on image data and labels and classification of images from the MNIST database consists of the following steps_:

1. Header files required for the neural network execution are included using the `#include` directive.
2. The `run_model()` function is implemented that:
    1. Calls `process_dataset()` to load and prepare MNIST data.
    2. Builds the neural network using `construct_network()` for the specified neuron type.
    3. Runs the complete training process with `train_model()`.
    4. Saves the trained network if a saving path is specified.
    5. Performs inference with `infer_model()` and collects spike data.
    6. Evaluates inference results using `evaluate_results()`.
3. The `main()` function is implemented that:
    1. Processes command-line arguments with `parse_arguments()`.
    2. Validates parsed model description. 
    3. Prompts user to confirm parameters before execution.
    4. Routes execution to appropriate neuron model based on configuration.
    The `main` function provides a complete end-to-end workflow that demonstrates the full lifecycle of neural network development.

This design enables researchers and developers to easily experiment with different neuron models, training configurations, and evaluation metrics while maintaining a consistent, well-tested execution framework.


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

© 2026 AO Kaspersky Lab