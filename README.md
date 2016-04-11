# Apache Horn

The Apache Horn is an Apache Incubating project, a neuron-centric programming model and Sync and Async hybrid distributed training framework, supports both data and model parallelism for training large models with massive datasets. Unlike most systems having matrix approach to neural network training, Horn adopted the the neuron-centric model which enables training large-scale deep learning on highly scalable CPU cluster. In the future, we plan also to support GPU accelerations for heterogeneous devices.

## High Scalability

The Apache Horn is an Sync and Async hybrid distributed training framework. Within single BSP job, each task group works asynchronously using region barrier synchronization instead of global barrier synchronization, and trains large-scale neural network model using assigned data sets in synchronous way.

## Getting Involved

Horn is an open source volunteer project under the Apache Software Foundation. We encourage you to learn about the project and contribute your expertise.

