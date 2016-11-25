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

import org.apache.hadoop.io.FloatWritable;
import org.apache.hama.HamaConfiguration;
import org.apache.horn.core.Constants.TrainingMethod;
import org.apache.horn.core.HornJob;
import org.apache.horn.core.Neuron;
import org.apache.horn.core.RecurrentDropoutNeuron;
import org.apache.horn.core.RecurrentLayeredNeuralNetwork;
import org.apache.horn.core.Synapse;
import org.apache.horn.funcs.SoftMax;
import org.apache.horn.funcs.SquaredError;
import org.apache.horn.funcs.Tanh;

public class ExclusiveOrRecurrentMultiLayerPerceptron {

  public static class StandardNeuron extends
      Neuron<Synapse<FloatWritable, FloatWritable>> {

    @Override
    public void forward(Iterable<Synapse<FloatWritable, FloatWritable>> messages)
        throws IOException {
      float sum = 0;
      for (Synapse<FloatWritable, FloatWritable> m : messages) {
        sum += m.getInput() * m.getWeight();
      }
      this.feedforward(squashingFunction.apply(sum));
    }

    @Override
    public void backward(
        Iterable<Synapse<FloatWritable, FloatWritable>> messages)
        throws IOException {
      float delta = 0;

      if (!this.isDropped()) {
        for (Synapse<FloatWritable, FloatWritable> m : messages) {
          // Calculates error gradient for each neuron
          delta += (m.getDelta() * m.getWeight());

          // Weight corrections
          float weight = -this.getLearningRate() * m.getDelta()
              * this.getOutput() + this.getMomentumWeight() * m.getPrevWeight();
          this.push(weight);
        }
      }

      this.backpropagate(delta * squashingFunction.applyDerivative(getOutput()));
    }
  }

  public static HornJob createJob(HamaConfiguration conf, String modelPath,
      String inputPath, float learningRate, float momemtumWeight,
      float regularizationWeight, int features, int hu, int labels,
      int stepSize, int numOutCells, int miniBatch, int maxIteration)
          throws IOException, InstantiationException, IllegalAccessException {

    boolean isRecurrent = (stepSize == 1 ? false: true);

    HornJob job = new HornJob(
        conf, RecurrentLayeredNeuralNetwork.class, ExclusiveOrRecurrentMultiLayerPerceptron.class);
    job.setTrainingSetPath(inputPath);
    job.setModelPath(modelPath);

    job.setMaxIteration(maxIteration);
    job.setLearningRate(learningRate);
    job.setMomentumWeight(momemtumWeight);
    job.setRegularizationWeight(regularizationWeight);

    job.setConvergenceCheckInterval(10);
    job.setBatchSize(miniBatch);
    job.setRecurrentStepSize(stepSize);

    job.setTrainingMethod(TrainingMethod.GRADIENT_DESCENT);

    job.inputLayer(features, 1.0f); // droprate
    job.addLayer(hu, Tanh.class, RecurrentDropoutNeuron.class, isRecurrent);
    job.addLayer(hu, Tanh.class, RecurrentDropoutNeuron.class, isRecurrent);
    job.outputLayer(labels, SoftMax.class, RecurrentDropoutNeuron.class, numOutCells);

    job.setCostFunction(SquaredError.class);

    return job;
  }

  public static void main(String[] args) throws IOException,
      InterruptedException, ClassNotFoundException,
      NumberFormatException, InstantiationException, IllegalAccessException {
    if (args.length < 9) {
      System.out.println("Usage: <MODEL_PATH> <INPUT_PATH> "
          + "<LEARNING_RATE> <MOMEMTUM_WEIGHT> <REGULARIZATION_WEIGHT> "
          + "<FEATURE_DIMENSION> <HIDDEN_UNITS> <LABEL_DIMENSION> "
          + "<STEP_SIZE> <NUM_OUTPUTCELLS> <BATCH_SIZE> <MAX_ITERATION>");
      System.out.println("E.g. MnistRecurrentMultiLayerPerceptron"
          + " ./model_semi_rnn semi.seq 0.01 0.9 0.0005 1 2 2 2 1 10 10000");
      System.exit(-1);
    }

    HornJob rnn = createJob(new HamaConfiguration(), args[0], args[1],
        Float.parseFloat(args[2]), Float.parseFloat(args[3]),
        Float.parseFloat(args[4]), Integer.parseInt(args[5]),
        Integer.parseInt(args[6]), Integer.parseInt(args[7]),
        Integer.parseInt(args[8]), Integer.parseInt(args[9]),
        Integer.parseInt(args[10]), Integer.parseInt(args[11]));

    long startTime = System.currentTimeMillis();
    if (rnn.waitForCompletion(true)) {
      System.out.println("Optimization Finished! "
          + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
    }
  }
}
