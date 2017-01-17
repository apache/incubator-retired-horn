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

import org.apache.hama.commons.math.FloatVector;
import org.apache.horn.core.Neuron;
import org.apache.horn.utils.MathUtils;

public class RecurrentDropoutNeuron extends Neuron {

  private float m2;
  private float recurrentDelta = 0;
  private double dropRate = 0;

  public double getDropRate() {
    return dropRate;
  }

  public void setDropRate(double dropRate) {
    this.dropRate = dropRate;
  }

  @Override
  public void forward(FloatVector inputVector) throws IOException {
    m2 = (isTraining()) ? MathUtils.getBinomial(1, dropRate) : 1.0f;

    if (m2 > 0) {
      float sum = inputVector.multiply(getWeightVector()).sum();
      setDrop(false);
      feedforward(squashingFunction.apply(sum) * m2);
    } else {
      setDrop(true);
      feedforward(0);
    }
  }

  @Override
  public void backward(FloatVector deltaVector) throws IOException {
    if (!this.isDropped()) {
      float delta = getWeightVector().multiply(deltaVector).sum();

      // update weights
      pushUpdates(deltaVector.multiply(-getLearningRate() * getOutput()).add(
          getPrevWeightVector().multiply(getMomentumWeight())));

      // TODO set squashingFunction of recurrent neurons identity
      backpropagate(recurrentDelta + delta
          * squashingFunction.applyDerivative(getOutput()));
    } else {
      backpropagate(0);
    }
  }

  public float getRecurrentDelta() {
    return recurrentDelta;
  }

  public void setRecurrentDelta(float recurrentDelta) {
    this.recurrentDelta = recurrentDelta;
  }
}
