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

import java.io.IOException;
import java.util.Map;

import org.apache.hadoop.fs.Path;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.BSPJob;
import org.apache.hama.commons.math.DenseFloatVector;
import org.apache.hama.commons.math.FloatFunction;
import org.apache.hama.commons.math.FloatMatrix;
import org.apache.hama.commons.math.FloatVector;
import org.apache.horn.core.Constants.LearningStyle;
import org.apache.horn.funcs.FunctionFactory;

import com.google.common.base.Preconditions;

/**
 * AutoEncoder is a model used for dimensional reduction and feature learning.
 * It is a special kind of {@link AbstractNeuralNetwork} that consists of three layers
 * of neurons, where the first layer and third layer contains the same number of
 * neurons.
 * 
 */
public class AutoEncoder {

  private final LayeredNeuralNetwork model;

  /**
   * Initialize the autoencoder.
   * 
   * @param inputDimensions The number of dimensions for the input feature.
   * @param compressedDimensions The number of dimensions for the compressed
   *          information.
   */
  public AutoEncoder(int inputDimensions, int compressedDimensions) {
    model = new LayeredNeuralNetwork();
    model.addLayer(inputDimensions, false,
        FunctionFactory.createFloatFunction("Sigmoid"), null);
    model.addLayer(compressedDimensions, false,
        FunctionFactory.createFloatFunction("Sigmoid"), null);
    model.addLayer(inputDimensions, true,
        FunctionFactory.createFloatFunction("Sigmoid"), null);
    model
        .setLearningStyle(LearningStyle.UNSUPERVISED);
    model.setCostFunction(FunctionFactory
        .createFloatFloatFunction("SquaredError"));
  }

  public AutoEncoder(HamaConfiguration conf, String modelPath) {
    model = new LayeredNeuralNetwork(conf, modelPath);
  }

  public AutoEncoder setModelPath(String modelPath) {
    model.setModelPath(modelPath);
    return this;
  }

  /**
   * Train the autoencoder with given data. Note that the training data is
   * pre-processed, where the features
   * 
   * @param dataInputPath
   * @param trainingParams
   * @throws InterruptedException 
   * @throws IOException 
   * @throws ClassNotFoundException 
   */
  public BSPJob train(HamaConfiguration conf, Path dataInputPath,
      Map<String, String> trainingParams) throws ClassNotFoundException, IOException, InterruptedException {
    return model.train(conf);
  }

  /**
   * Train the model with one instance.
   * 
   * @param trainingInstance
   */
  public void trainOnline(FloatVector trainingInstance) {
    model.trainOnline(trainingInstance);
  }

  /**
   * Get the matrix M used to encode the input features.
   * 
   * @return this matrix with encode the input.
   */
  public FloatMatrix getEncodeWeightMatrix() {
    return model.getWeightsByLayer(0);
  }

  /**
   * Get the matrix M used to decode the compressed information.
   * 
   * @return this matrix with decode the compressed information.
   */
  public FloatMatrix getDecodeWeightMatrix() {
    return model.getWeightsByLayer(1);
  }

  /**
   * Transform the input features.
   * 
   * @param inputInstance
   * @return The compressed information.
   */
  private FloatVector transform(FloatVector inputInstance, int inputLayer) {
    FloatVector internalInstance = new DenseFloatVector(
        inputInstance.getDimension() + 1);
    internalInstance.set(0, 1);
    for (int i = 0; i < inputInstance.getDimension(); ++i) {
      internalInstance.set(i + 1, inputInstance.get(i));
    }
    FloatFunction squashingFunction = model.getSquashingFunction(inputLayer);
    FloatMatrix weightMatrix = null;
    if (inputLayer == 0) {
      weightMatrix = this.getEncodeWeightMatrix();
    } else {
      weightMatrix = this.getDecodeWeightMatrix();
    }
    FloatVector vec = weightMatrix.multiplyVectorUnsafe(internalInstance);
    vec = vec.applyToElements(squashingFunction);
    return vec;
  }

  /**
   * Encode the input instance.
   * 
   * @param inputInstance
   * @return a new vector with the encode input instance.
   */
  public FloatVector encode(FloatVector inputInstance) {
    Preconditions
        .checkArgument(
            inputInstance.getDimension() == model.getLayerSize(0) - 1,
            String
                .format(
                    "The dimension of input instance is %d, but the model requires dimension %d.",
                    inputInstance.getDimension(), model.getLayerSize(1) - 1));
    return this.transform(inputInstance, 0);
  }

  /**
   * Decode the input instance.
   * 
   * @param inputInstance
   * @return a new vector with the decode input instance.
   */
  public FloatVector decode(FloatVector inputInstance) {
    Preconditions
        .checkArgument(
            inputInstance.getDimension() == model.getLayerSize(1) - 1,
            String
                .format(
                    "The dimension of input instance is %d, but the model requires dimension %d.",
                    inputInstance.getDimension(), model.getLayerSize(1) - 1));
    return this.transform(inputInstance, 1);
  }

  /**
   * Get the label(s) according to the given features.
   * 
   * @param inputInstance
   * @return a new vector with output of the model according to given feature
   *         instance.
   */
  public FloatVector getOutput(FloatVector inputInstance) {
    return model.getOutput(inputInstance);
  }

  /**
   * Set the feature transformer.
   * 
   * @param featureTransformer
   */
  public void setFeatureTransformer(FloatFeatureTransformer featureTransformer) {
    this.model.setFeatureTransformer(featureTransformer);
  }

}
