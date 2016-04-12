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

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.TextInputFormat;
import org.apache.horn.bsp.HornJob;
import org.apache.horn.funcs.CrossEntropy;
import org.apache.horn.funcs.Sigmoid;
import org.apache.horn.trainer.Neuron;
import org.apache.horn.trainer.PropMessage;

public class MultiLayerPerceptron {

  public static class StandardNeuron extends
      Neuron<PropMessage<DoubleWritable, DoubleWritable>> {

    private double learningRate;
    private double lambda;
    private double momentum;
    private static double bias = -1;

    @Override
    public void setup(HamaConfiguration conf) {
      this.learningRate = conf.getDouble("mlp.learning.rate", 0.1);
      this.lambda = conf.getDouble("mlp.regularization.weight", 0.01);
      this.momentum = conf.getDouble("mlp.momentum.weight", 0.2);
    }

    @Override
    public void forward(
        Iterable<PropMessage<DoubleWritable, DoubleWritable>> messages)
        throws IOException {
      double sum = 0;

      for (PropMessage<DoubleWritable, DoubleWritable> m : messages) {
        sum += m.getInput() * m.getWeight();
      }
      sum += bias * this.getTheta(); // add bias feature
      feedforward(activation(sum));
    }

    @Override
    public void backward(
        Iterable<PropMessage<DoubleWritable, DoubleWritable>> messages)
        throws IOException {
      for (PropMessage<DoubleWritable, DoubleWritable> m : messages) {
        // Calculates error gradient for each neuron
        double gradient = this.getOutput() * (1 - this.getOutput())
            * m.getDelta() * m.getWeight();
        backpropagate(gradient);

        // Weight corrections
        double weight = -learningRate * this.getOutput() * m.getDelta()
            + momentum * this.getPreviousWeight();
        this.push(weight);
      }
    }

  }

  public static void main(String[] args) throws IOException,
      InterruptedException, ClassNotFoundException {
    HamaConfiguration conf = new HamaConfiguration();
    HornJob job = new HornJob(conf, MultiLayerPerceptron.class);

    job.setDouble("mlp.learning.rate", 0.1);
    job.setDouble("mlp.regularization.weight", 0.01);
    job.setDouble("mlp.momentum.weight", 0.2);

    // initialize the topology of the model.
    // a three-layer model is created in this example
    job.addLayer(1000, StandardNeuron.class, Sigmoid.class); // 1st layer
    job.addLayer(800, StandardNeuron.class, Sigmoid.class); // 2nd layer
    job.addLayer(300, StandardNeuron.class, Sigmoid.class); // total classes

    // set the cost function to evaluate the error
    job.setCostFunction(CrossEntropy.class);

    // set I/O and others
    job.setInputFormat(TextInputFormat.class);
    job.setOutputPath(new Path("/tmp/"));
    job.setMaxIteration(10000);
    job.setNumBspTask(3);

    long startTime = System.currentTimeMillis();

    if (job.waitForCompletion(true)) {
      System.out.println("Job Finished in "
          + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
    }
  }
}
