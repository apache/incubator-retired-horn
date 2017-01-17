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
package org.apache.horn.core;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hama.commons.math.FloatFunction;
import org.apache.hama.commons.math.FloatVector;

public abstract class Neuron implements Writable, NeuronInterface {
  int id;
  float output;
  float weight;
  float delta;

  float momentumWeight;
  float learningRate;

  int layerIndex;
  boolean isOutputLayer;
  boolean isTraining;
  boolean isDropped;
  long iterations;

  protected FloatFunction squashingFunction;

  public void setNeuronID(int id) {
    this.id = id;
  }

  public int getNeuronID() {
    return id;
  }

  public int getLayerIndex() {
    return layerIndex;
  }

  public void setLayerIndex(int index) {
    this.layerIndex = index;
  }

  public void feedforward(float sum) {
    this.output = sum;
  }

  public void backpropagate(float gradient) {
    this.delta = gradient;
  }

  public float getDelta() {
    return delta;
  }

  public void setWeight(float weight) {
    this.weight = weight;
  }

  public void setOutput(float output) {
    this.output = output;
  }

  public float getOutput() {
    return output;
  }

  public void setMomentumWeight(float momentumWeight) {
    this.momentumWeight = momentumWeight;
  }

  public float getMomentumWeight() {
    return momentumWeight;
  }

  public void setLearningRate(float learningRate) {
    this.learningRate = learningRate;
  }

  public float getLearningRate() {
    return learningRate;
  }

  // ////////
  float[] weights;

  public void setWeightVector(int rowCount) {
    weights = new float[rowCount];
  }

  public float[] getUpdates() {
    return weights;
  }

  public void pushUpdates(FloatVector weights) {
    this.weights = weights.toArray();
  }

  public void setSquashingFunction(FloatFunction squashingFunction) {
    this.squashingFunction = squashingFunction;
  }

  public void setTraining(boolean b) {
    this.isTraining = b;
  }

  public boolean isTraining() {
    return isTraining;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    id = in.readInt();
    output = in.readFloat();
    weight = in.readFloat();
    delta = in.readFloat();
    iterations = in.readLong();

    momentumWeight = in.readFloat();
    learningRate = in.readFloat();
    isTraining = in.readBoolean();
    isDropped = in.readBoolean();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(id);
    out.writeFloat(output);
    out.writeFloat(weight);
    out.writeFloat(delta);
    out.writeLong(iterations);

    out.writeFloat(momentumWeight);
    out.writeFloat(learningRate);
    out.writeBoolean(isTraining);
    out.writeBoolean(isDropped);
  }

  public void setIterationNumber(long iterations) {
    this.iterations = iterations;
  }

  public long getIterationNumber() {
    return iterations;
  }

  public boolean isDropped() {
    return isDropped;
  }

  public void setDrop(boolean isDropped) {
    this.isDropped = isDropped;
  }

  private float nablaW;

  public void setNablaW(float f) {
    // TODO Auto-generated method stub
    nablaW = f;
  }

  public float getNablaW() {
    return nablaW;
  }

  
  
  
  /////
  private FloatVector weightVector;
  private FloatVector deltaVector;
  private FloatVector prevWeightVector;

  public void setWeightVector(FloatVector weightVector) {
    this.weightVector = weightVector;
  }

  public FloatVector getWeightVector() {
    return weightVector;
  }

  public void setDeltaVector(FloatVector deltaVector) {
    this.deltaVector = deltaVector;
  }

  public FloatVector getDeltaVector() {
    return deltaVector;
  }

  public void setPrevWeightVector(FloatVector prevWeightVector) {
    this.prevWeightVector = prevWeightVector;
  }

  public FloatVector getPrevWeightVector() {
    return prevWeightVector;
  }

}
