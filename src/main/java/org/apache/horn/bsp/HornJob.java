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
package org.apache.horn.bsp;

import java.io.IOException;

import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.BSPJob;
import org.apache.hama.commons.math.Function;
import org.apache.hama.commons.math.FunctionFactory;

public class HornJob extends BSPJob {

  SmallLayeredNeuralNetwork neuralNetwork;

  public HornJob(HamaConfiguration conf, Class<?> exampleClass)
      throws IOException {
    super(conf);
    this.setJarByClass(exampleClass);

    neuralNetwork = new SmallLayeredNeuralNetwork();
  }

  public void addLayer(int featureDimension, Class<? extends Function> func) {
    neuralNetwork.addLayer(featureDimension, false,
        FunctionFactory.createDoubleFunction(func.getSimpleName()));
  }

  public void finalLayer(int labels, Class<? extends Function> func) {
    neuralNetwork.addLayer(labels, true,
        FunctionFactory.createDoubleFunction(func.getSimpleName()));
  }

  public void setCostFunction(Class<? extends Function> func) {
    neuralNetwork.setCostFunction(FunctionFactory
        .createDoubleDoubleFunction(func.getSimpleName()));
  }

  public void setDouble(String name, double value) {
    conf.setDouble(name, value);
  }

  public void setMaxIteration(int maxIteration) {
    this.conf.setInt("training.max.iterations", maxIteration);
  }

  public void setBatchSize(int batchSize) {
    this.conf.setInt("training.batch.size", batchSize);
  }

  public void setLearningRate(double learningRate) {
    this.conf.setDouble("mlp.learning.rate", learningRate);
  }

  public void setConvergenceCheckInterval(int n) {
    this.conf.setInt("convergence.check.interval", n);
  }

  public void setMomentumWeight(double momentumWeight) {
    this.conf.setDouble("mlp.momentum.weight", momentumWeight);
  }

  public SmallLayeredNeuralNetwork getNeuralNetwork() {
    return neuralNetwork;
  }

  public boolean waitForCompletion(boolean verbose) throws IOException,
      InterruptedException, ClassNotFoundException {
    BSPJob job = neuralNetwork.train(this.conf);
    if (verbose) {
      return job.waitForCompletion(true);
    } else {
      return job.waitForCompletion(false);
    }
  }

  public void setRegularizationWeight(double regularizationWeight) {
    this.conf.setDouble("regularization.weight", regularizationWeight);
  }

  public void setModelPath(String modelPath) {
    this.conf.set("model.path", modelPath);
    neuralNetwork.setModelPath(modelPath);
  }

  public void setTrainingSetPath(String inputPath) {
    this.conf.set("training.input.path", inputPath);
  }

}
