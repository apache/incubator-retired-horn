/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.horn.examples;

import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hama.HamaConfiguration;
import org.apache.horn.bsp.HornJob;
import org.apache.horn.bsp.Neuron;
import org.apache.horn.bsp.StandardNeuron;
import org.apache.horn.bsp.ConvNeuron;
import org.apache.horn.bsp.Synapse;
import org.apache.horn.funcs.CrossEntropy;
import org.apache.horn.funcs.Sigmoid;

public class ConvolutionalNeuralNetwork {

  //possible way to pass classes is have a list of tuples to pass the neuron and squashing funciton value at the same time
  public static CNNHornJob createJob(HamaConfiguration conf, String modelPath,
      String inputPath, double learningRate, double momemtumWeight,
      double regularizationWeight, int features, int labels, int maxIteration,
      int numOfTasks) throws IOException {

    CNNHornJob job = new CNNHornJob(conf, ConvolutionalNeuralNetwork.class);
    job.setTrainingSetPath(inputPath);
    job.setModelPath(modelPath);

    job.setNumBspTask(numOfTasks);
    job.setMaxIteration(maxIteration);
    job.setLearningRate(learningRate);
    job.setMomentumWeight(momemtumWeight);
    job.setRegularizationWeight(regularizationWeight);

    job.setConvergenceCheckInterval(1000);
    job.setBatchSize(300);

    cnn.addLayer(150, ReLu.class, ConvNeuron.class); // convolution layer
    ccn.addLayer(100, Sigmoid.class, StandardNeuron.class); // fully connected
    ccn.addLayer(100, Sigmoid.class, StandardNeuron.class); // fully connected
    ccn.outputLayer(10, Sigmoid.class, StandardNeuron.class); // fully connected

    job.setCostFunction(CrossEntropy.class);

    return job;
  }

  public static void main(String[] args) throws IOException,
      InterruptedException, ClassNotFoundException {
    //TODO: implement this for real
    if (args.length < 9) {
      System.out
          .println("Usage: model_path training_set learning_rate momentum regularization_weight feature_dimension label_dimension max_iteration num_tasks");
      System.exit(1);
    }
    HornJob ann = createJob(new HamaConfiguration(), args[0], args[1],
        Double.parseDouble(args[2]), Double.parseDouble(args[3]),
        Double.parseDouble(args[4]), Integer.parseInt(args[5]),
        Integer.parseInt(args[6]), Integer.parseInt(args[7]),
        Integer.parseInt(args[8]));

    long startTime = System.currentTimeMillis();
    if (ann.waitForCompletion(true)) {
      System.out.println("Job Finished in "
          + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
    }
  }
}
