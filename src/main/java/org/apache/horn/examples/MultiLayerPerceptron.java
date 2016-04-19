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
import org.apache.horn.bsp.Synapse;
import org.apache.horn.funcs.CrossEntropy;
import org.apache.horn.funcs.Sigmoid;

public class MultiLayerPerceptron {

  public static class StandardNeuron extends
      Neuron<Synapse<DoubleWritable, DoubleWritable>> {
    private double learningRate;
    private double momentum;

    @Override
    public void setup(HamaConfiguration conf) {
      this.learningRate = conf.getDouble("mlp.learning.rate", 0.5);
      this.momentum = conf.getDouble("mlp.momentum.weight", 0.2);
    }

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

  public static HornJob createJob(HamaConfiguration conf, String modelPath,
      String inputPath, double learningRate, double momemtumWeight,
      double regularizationWeight, int features, int labels, int maxIteration,
      int numOfTasks) throws IOException {

    HornJob job = new HornJob(conf, MultiLayerPerceptron.class);
    job.setTrainingSetPath(inputPath);
    job.setModelPath(modelPath);

    job.setNumBspTask(numOfTasks);
    job.setMaxIteration(maxIteration);
    job.setLearningRate(learningRate);
    job.setMomentumWeight(momemtumWeight);
    job.setRegularizationWeight(regularizationWeight);

    job.setConvergenceCheckInterval(1000);
    job.setBatchSize(300);

    job.addLayer(features, Sigmoid.class);
    job.addLayer(features, Sigmoid.class);
    job.finalLayer(labels, Sigmoid.class);

    job.setCostFunction(CrossEntropy.class);

    return job;
  }

  public static void main(String[] args) throws IOException,
      InterruptedException, ClassNotFoundException {
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
