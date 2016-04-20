# Apache Horn

The Apache Horn is an Apache Incubating project, a neuron-centric programming model and Sync and Async hybrid distributed training framework, supports both data and model parallelism for training large models with massive datasets on top of Apache Hadoop and Hama.

## Programming Model

Apache Horn provides a neuron-centric programming model for implementing the neural network based algorithms. The user defines the computation that takes place at each neuron in each layer of the model, and the messages that should be passed during the forward and backward phases of computation. For example, we apply a set of weights to the input data and calculate an output in forward() method like below:
```Java
    @Override
    public void forward(
        Iterable<Synapse<DoubleWritable, DoubleWritable>> messages)
        throws IOException {
      double sum = 0;
      for (Synapse<DoubleWritable, DoubleWritable> m : messages) {
        sum += m.getInput() * m.getWeight();
      }
      this.feedforward(this.squashingFunction.apply(sum));
    }
```
Then, we measure the margin of error of the output and adjust the weights accordingly to decrease the error in backward() method:
```Java
    @Override
    public void backward(
        Iterable<Synapse<DoubleWritable, DoubleWritable>> messages)
        throws IOException {
      for (Synapse<DoubleWritable, DoubleWritable> m : messages) {
        // Calculates error gradient for each neuron
        double gradient = this.squashingFunction.applyDerivative(this
            .getOutput()) * (m.getDelta() * m.getWeight());
        this.backpropagate(gradient);

        // Weight corrections
        double weight = -learningRate * this.getOutput() * m.getDelta()
            + momentum * m.getPrevWeight();
        this.push(weight);
      }
    }
  }
```
The advantages of this programming model are:

 * Easy and intuitive to use
 * Flexible to make your own CUDA kernels
 * Allows multithreading to be used internally

Also, Apache Horn provides a simplified and intuitive configuration interface. To create neural network job and submit it to existing Hadoop or Hama cluster, we just add the layer with its properties such as squashing function and neuron class. The below example configures the create 4-layer neural network with 500 neurons in hidden layers for train MNIST dataset:
```Java
  HornJob job = new HornJob(conf, MultiLayerPerceptron.class);
  job.setLearningRate(learningRate);
  ..

  job.inputLayer(784, Sigmoid.class, StandardNeuron.class);
  job.addLayer(500, Sigmoid.class, StandardNeuron.class);
  job.addLayer(500, Sigmoid.class, StandardNeuron.class);
  job.outputLayer(10, Sigmoid.class, StandardNeuron.class);
  job.setCostFunction(CrossEntropy.class);
```

## High Scalability

The Apache Horn is an Sync and Async hybrid distributed training framework. Within single BSP job, each task group works asynchronously using region barrier synchronization instead of global barrier synchronization, and trains large-scale neural network model using assigned data sets in synchronous way.

## Getting Involved

Horn is an open source volunteer project under the Apache Software Foundation. We encourage you to learn about the project and contribute your expertise.

