# Apache Horn

The Apache Horn is an Apache Incubating project, a neuron-centric programming model and Sync and Async hybrid distributed training framework, supports both data and model parallelism for training large models with massive datasets. Unlike most systems having matrix approach to neural network training, Horn adopted the the neuron-centric model which enables training large-scale deep learning on highly scalable CPU cluster. In the future, we plan also to support GPU accelerations for heterogeneous devices.

## Tensor vs. Neuron

While tensor-based models would require an large memory consumption or parallel computational complexity to calibrate a large number of model parameters, the neuron-centric model has advantages like below:
 
 * More intuitive programming APIs
 * An effective partition and parallelization strategy for large model
 * Easy to understand how groups of neurons communicate 

|             | Tensor           | Neuron  |
| ------------- |:-------------:|:-----:|
| Computation model	| tensor/matrix-based computation model | neuron-based iterative computation model |
| Partitioning models | Vector or Submatrix (block) | Subgraph components (densely connected areas) |
| Communication overhead | Large |  Small |

## High Scalability

The Apache Horn is an Sync and Async hybrid distributed training framework. Within single BSP job, each task group works asynchronously using region barrier synchronization instead of global barrier synchronization, and trains large-scale neural network model using assigned data sets in synchronous way.

## Getting Involved

Horn is an open source volunteer project under the Apache Software Foundation. We encourage you to learn about the project and contribute your expertise.
