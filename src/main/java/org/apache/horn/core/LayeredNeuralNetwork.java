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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.math.RandomUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.BSPJob;
import org.apache.hama.Constants;
import org.apache.hama.commons.io.MatrixWritable;
import org.apache.hama.commons.io.VectorWritable;
import org.apache.hama.commons.math.DenseDoubleMatrix;
import org.apache.hama.commons.math.DenseDoubleVector;
import org.apache.hama.commons.math.DoubleFunction;
import org.apache.hama.commons.math.DoubleMatrix;
import org.apache.hama.commons.math.DoubleVector;
import org.apache.hama.util.ReflectionUtils;
import org.apache.horn.core.Constants.LearningStyle;
import org.apache.horn.core.Constants.TrainingMethod;
import org.apache.horn.examples.MultiLayerPerceptron.StandardNeuron;
import org.apache.horn.funcs.FunctionFactory;
import org.apache.horn.funcs.IdentityFunction;
import org.apache.horn.funcs.SoftMax;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

/**
 * SmallLayeredNeuralNetwork defines the general operations for derivative
 * layered models, include Linear Regression, Logistic Regression, Multilayer
 * Perceptron, Autoencoder, and Restricted Boltzmann Machine, etc. For
 * SmallLayeredNeuralNetwork, the training can be conducted in parallel, but the
 * parameters of the models are assumes to be stored in a single machine.
 * 
 * In general, these models consist of neurons which are aligned in layers.
 * Between layers, for any two adjacent layers, the neurons are connected to
 * form a bipartite weighted graph.
 * 
 */
public class LayeredNeuralNetwork extends AbstractLayeredNeuralNetwork {

  private static final Log LOG = LogFactory.getLog(LayeredNeuralNetwork.class);

  /* Weights between neurons at adjacent layers */
  protected List<DoubleMatrix> weightMatrixList;

  /* Previous weight updates between neurons at adjacent layers */
  protected List<DoubleMatrix> prevWeightUpdatesList;

  /* Different layers can have different squashing function */
  protected List<DoubleFunction> squashingFunctionList;

  protected List<Class<? extends Neuron>> neuronClassList;

  protected int finalLayerIdx;

  private List<Neuron[]> neurons = new ArrayList<Neuron[]>();

  public LayeredNeuralNetwork() {
    this.layerSizeList = Lists.newArrayList();
    this.weightMatrixList = Lists.newArrayList();
    this.prevWeightUpdatesList = Lists.newArrayList();
    this.squashingFunctionList = Lists.newArrayList();
    this.neuronClassList = Lists.newArrayList();
  }

  public LayeredNeuralNetwork(HamaConfiguration conf, String modelPath) {
    super(conf, modelPath);

    // initialize neuron objects
    for (int i = 0; i < layerSizeList.size(); i++) {
      int numOfNeurons = layerSizeList.get(i);
      Class neuronClass;
      if (i == 0)
        neuronClass = Neuron.class;
      else
        neuronClass = neuronClassList.get(i - 1);

      Neuron[] tmp = new Neuron[numOfNeurons];
      for (int j = 0; j < numOfNeurons; j++) {
        Neuron n = newNeuronInstance(StandardNeuron.class);
        if (i > 0)
          n.setSquashingFunction(squashingFunctionList.get(i - 1));
        else
          n.setSquashingFunction(new IdentityFunction());

        n.setLayerIndex(i);

        n.setLearningRate(this.learningRate);
        n.setMomentumWeight(this.momentumWeight);
        tmp[j] = n;
      }

      neurons.add(tmp);
    }
  }

  @Override
  /**
   * {@inheritDoc}
   */
  public int addLayer(int size, boolean isFinalLayer,
      DoubleFunction squashingFunction, Class<? extends Neuron> neuronClass) {
    return addLayer(size, isFinalLayer, squashingFunction, neuronClass, null);
  }

  public int addLayer(int size, boolean isFinalLayer,
      DoubleFunction squashingFunction, Class<? extends Neuron> neuronClass,
      Class<? extends IntermediateOutput> interlayer) {
    Preconditions.checkArgument(size > 0,
        "Size of layer must be larger than 0.");
    if (!isFinalLayer) {
      if (this.layerSizeList.size() == 0) {
        LOG.info("add input layer: " + size + " neurons");
      } else {
        LOG.info("add hidden layer: " + size + " neurons");
      }
      size += 1;
    }

    this.layerSizeList.add(size);
    int layerIdx = this.layerSizeList.size() - 1;
    if (isFinalLayer) {
      this.finalLayerIdx = layerIdx;
      LOG.info("add output layer: " + size + " neurons");
    }

    // add weights between current layer and previous layer, and input layer has
    // no squashing function
    if (layerIdx > 0) {
      int sizePrevLayer = this.layerSizeList.get(layerIdx - 1);
      // row count equals to size of current size and column count equals to
      // size of previous layer
      int row = isFinalLayer ? size : size - 1;
      int col = sizePrevLayer;
      DoubleMatrix weightMatrix = new DenseDoubleMatrix(row, col);
      // initialize weights
      weightMatrix.applyToElements(new DoubleFunction() {
        @Override
        public double apply(double value) {
          return RandomUtils.nextDouble() - 0.5;
        }

        @Override
        public double applyDerivative(double value) {
          throw new UnsupportedOperationException("");
        }
      });
      this.weightMatrixList.add(weightMatrix);
      this.prevWeightUpdatesList.add(new DenseDoubleMatrix(row, col));
      this.squashingFunctionList.add(squashingFunction);

      this.neuronClassList.add(neuronClass);
    }
    return layerIdx;
  }

  /**
   * Update the weight matrices with given matrices.
   * 
   * @param matrices
   */
  public void updateWeightMatrices(DoubleMatrix[] matrices) {
    for (int i = 0; i < matrices.length; ++i) {
      DoubleMatrix matrix = this.weightMatrixList.get(i);
      this.weightMatrixList.set(i, matrix.add(matrices[i]));
    }
  }

  /**
   * Set the previous weight matrices.
   * 
   * @param prevUpdates
   */
  void setPrevWeightMatrices(DoubleMatrix[] prevUpdates) {
    this.prevWeightUpdatesList.clear();
    Collections.addAll(this.prevWeightUpdatesList, prevUpdates);
  }

  /**
   * Add a batch of matrices onto the given destination matrices.
   * 
   * @param destMatrices
   * @param sourceMatrices
   */
  static void matricesAdd(DoubleMatrix[] destMatrices,
      DoubleMatrix[] sourceMatrices) {
    for (int i = 0; i < destMatrices.length; ++i) {
      destMatrices[i] = destMatrices[i].add(sourceMatrices[i]);
    }
  }

  /**
   * Get all the weight matrices.
   * 
   * @return The matrices in form of matrix array.
   */
  DoubleMatrix[] getWeightMatrices() {
    DoubleMatrix[] matrices = new DoubleMatrix[this.weightMatrixList.size()];
    this.weightMatrixList.toArray(matrices);
    return matrices;
  }

  /**
   * Set the weight matrices.
   * 
   * @param matrices
   */
  public void setWeightMatrices(DoubleMatrix[] matrices) {
    this.weightMatrixList = new ArrayList<DoubleMatrix>();
    Collections.addAll(this.weightMatrixList, matrices);
  }

  /**
   * Get the previous matrices updates in form of array.
   * 
   * @return The matrices in form of matrix array.
   */
  public DoubleMatrix[] getPrevMatricesUpdates() {
    DoubleMatrix[] prevMatricesUpdates = new DoubleMatrix[this.prevWeightUpdatesList
        .size()];
    for (int i = 0; i < this.prevWeightUpdatesList.size(); ++i) {
      prevMatricesUpdates[i] = this.prevWeightUpdatesList.get(i);
    }
    return prevMatricesUpdates;
  }

  public void setWeightMatrix(int index, DoubleMatrix matrix) {
    Preconditions.checkArgument(
        0 <= index && index < this.weightMatrixList.size(), String.format(
            "index [%d] should be in range[%d, %d].", index, 0,
            this.weightMatrixList.size()));
    this.weightMatrixList.set(index, matrix);
  }

  @Override
  public void readFields(DataInput input) throws IOException {
    super.readFields(input);

    this.finalLayerIdx = input.readInt();

    // read neuron classes
    int neuronClasses = input.readInt();
    this.neuronClassList = Lists.newArrayList();
    for (int i = 0; i < neuronClasses; ++i) {
      try {
        Class<? extends Neuron> clazz = (Class<? extends Neuron>) Class
            .forName(input.readUTF());
        neuronClassList.add(clazz);
      } catch (ClassNotFoundException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
    }

    // read squash functions
    int squashingFunctionSize = input.readInt();
    this.squashingFunctionList = Lists.newArrayList();
    for (int i = 0; i < squashingFunctionSize; ++i) {
      this.squashingFunctionList.add(FunctionFactory
          .createDoubleFunction(WritableUtils.readString(input)));
    }

    // read weights and construct matrices of previous updates
    int numOfMatrices = input.readInt();
    this.weightMatrixList = Lists.newArrayList();
    this.prevWeightUpdatesList = Lists.newArrayList();
    for (int i = 0; i < numOfMatrices; ++i) {
      DoubleMatrix matrix = MatrixWritable.read(input);
      this.weightMatrixList.add(matrix);
      this.prevWeightUpdatesList.add(new DenseDoubleMatrix(
          matrix.getRowCount(), matrix.getColumnCount()));
    }

  }

  @Override
  public void write(DataOutput output) throws IOException {
    super.write(output);

    output.writeInt(finalLayerIdx);

    // write neuron classes
    output.writeInt(this.neuronClassList.size());
    for (Class<? extends Neuron> clazz : this.neuronClassList) {
      output.writeUTF(clazz.getName());
    }

    // write squashing functions
    output.writeInt(this.squashingFunctionList.size());
    for (DoubleFunction aSquashingFunctionList : this.squashingFunctionList) {
      WritableUtils.writeString(output,
          aSquashingFunctionList.getFunctionName());
    }

    // write weight matrices
    output.writeInt(this.weightMatrixList.size());
    for (DoubleMatrix aWeightMatrixList : this.weightMatrixList) {
      MatrixWritable.write(aWeightMatrixList, output);
    }

    // DO NOT WRITE WEIGHT UPDATE
  }

  @Override
  public DoubleMatrix getWeightsByLayer(int layerIdx) {
    return this.weightMatrixList.get(layerIdx);
  }

  /**
   * Get the output of the model according to given feature instance.
   */
  @Override
  public DoubleVector getOutput(DoubleVector instance) {
    Preconditions.checkArgument(this.layerSizeList.get(0) - 1 == instance
        .getDimension(), String.format(
        "The dimension of input instance should be %d.",
        this.layerSizeList.get(0) - 1));
    // transform the features to another space
    DoubleVector transformedInstance = this.featureTransformer
        .transform(instance);
    // add bias feature
    DoubleVector instanceWithBias = new DenseDoubleVector(
        transformedInstance.getDimension() + 1);
    instanceWithBias.set(0, 0.99999); // set bias to be a little bit less than
                                      // 1.0
    for (int i = 1; i < instanceWithBias.getDimension(); ++i) {
      instanceWithBias.set(i, transformedInstance.get(i - 1));
    }

    // return the output of the last layer
    return getOutputInternal(instanceWithBias);
  }

  /**
   * Calculate output internally, the intermediate output of each layer will be
   * stored.
   * 
   * @param instanceWithBias The instance contains the features.
   * @return Cached output of each layer.
   */
  public DoubleVector getOutputInternal(DoubleVector instanceWithBias) {
    // sets the output of input layer
    Neuron[] inputLayer = neurons.get(0);
    for (int i = 0; i < inputLayer.length; i++) {
      inputLayer[i].setOutput(instanceWithBias.get(i));
    }

    for (int i = 0; i < this.layerSizeList.size() - 1; ++i) {
      forward(i);
    }

    DoubleVector output = new DenseDoubleVector(
        neurons.get(this.finalLayerIdx).length);
    for (int i = 0; i < output.getDimension(); i++) {
      output.set(i, neurons.get(this.finalLayerIdx)[i].getOutput());
    }

    return output;
  }

  /**
   * @param neuronClass
   * @return a new neuron instance
   */
  @SuppressWarnings({ "rawtypes" })
  public static Neuron newNeuronInstance(Class<? extends Neuron> neuronClass) {
    return (Neuron) ReflectionUtils.newInstance(neuronClass);
  }

  /**
   * Forward the calculation for one layer.
   * 
   * @param fromLayer The index of the previous layer.
   */
  protected void forward(int fromLayer) {
    int curLayerIdx = fromLayer + 1;
    DoubleMatrix weightMatrix = this.weightMatrixList.get(fromLayer);

    DoubleFunction squashingFunction = getSquashingFunction(fromLayer);
    DoubleVector vec = new DenseDoubleVector(weightMatrix.getRowCount());

    for (int row = 0; row < weightMatrix.getRowCount(); row++) {
      List<Synapse<DoubleWritable, DoubleWritable>> msgs = new ArrayList<Synapse<DoubleWritable, DoubleWritable>>();
      for (int col = 0; col < weightMatrix.getColumnCount(); col++) {
        msgs.add(new Synapse<DoubleWritable, DoubleWritable>(
            new DoubleWritable(neurons.get(fromLayer)[col].getOutput()),
            new DoubleWritable(weightMatrix.get(row, col))));
      }

      Neuron n;
      if (curLayerIdx == finalLayerIdx)
        n = neurons.get(curLayerIdx)[row];
      else
        n = neurons.get(curLayerIdx)[row + 1];

      try {
        n.forward(msgs);
      } catch (IOException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
      vec.set(row, n.getOutput());
    }

    if (squashingFunction.getFunctionName().equalsIgnoreCase(
        SoftMax.class.getSimpleName())) {
      IntermediateOutput interlayer = (IntermediateOutput) ReflectionUtils
          .newInstance(SoftMax.SoftMaxOutputComputer.class);
      try {
        vec = interlayer.interlayer(vec);

        for (int i = 0; i < vec.getDimension(); i++) {
          neurons.get(curLayerIdx)[i].setOutput(vec.get(i));
        }

      } catch (IOException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
    }

    // add bias
    if (curLayerIdx != finalLayerIdx)
      neurons.get(curLayerIdx)[0].setOutput(1);
  }

  /**
   * Train the model online.
   * 
   * @param trainingInstance
   */
  public void trainOnline(DoubleVector trainingInstance) {
    DoubleMatrix[] updateMatrices = this.trainByInstance(trainingInstance);
    this.updateWeightMatrices(updateMatrices);
  }

  @Override
  public DoubleMatrix[] trainByInstance(DoubleVector trainingInstance) {
    DoubleVector transformedVector = this.featureTransformer
        .transform(trainingInstance.sliceUnsafe(this.layerSizeList.get(0) - 1));

    int inputDimension = this.layerSizeList.get(0) - 1;
    int outputDimension;
    DoubleVector inputInstance = null;
    DoubleVector labels = null;
    if (this.learningStyle == LearningStyle.SUPERVISED) {
      outputDimension = this.layerSizeList.get(this.layerSizeList.size() - 1);
      // validate training instance
      Preconditions.checkArgument(
          inputDimension + outputDimension == trainingInstance.getDimension(),
          String
              .format(
                  "The dimension of training instance is %d, but requires %d.",
                  trainingInstance.getDimension(), inputDimension
                      + outputDimension));

      inputInstance = new DenseDoubleVector(this.layerSizeList.get(0));
      inputInstance.set(0, 1); // add bias
      // get the features from the transformed vector
      for (int i = 0; i < inputDimension; ++i) {
        inputInstance.set(i + 1, transformedVector.get(i));
      }
      // get the labels from the original training instance
      labels = trainingInstance.sliceUnsafe(inputInstance.getDimension() - 1,
          trainingInstance.getDimension() - 1);
    } else if (this.learningStyle == LearningStyle.UNSUPERVISED) {
      // labels are identical to input features
      outputDimension = inputDimension;
      // validate training instance
      Preconditions.checkArgument(inputDimension == trainingInstance
          .getDimension(), String.format(
          "The dimension of training instance is %d, but requires %d.",
          trainingInstance.getDimension(), inputDimension));

      inputInstance = new DenseDoubleVector(this.layerSizeList.get(0));
      inputInstance.set(0, 1); // add bias
      // get the features from the transformed vector
      for (int i = 0; i < inputDimension; ++i) {
        inputInstance.set(i + 1, transformedVector.get(i));
      }
      // get the labels by copying the transformed vector
      labels = transformedVector.deepCopy();
    }

    DoubleVector output = this.getOutputInternal(inputInstance);

    // get the training error
    calculateTrainingError(labels, output);

    if (this.trainingMethod.equals(TrainingMethod.GRADIENT_DESCENT)) {
      return this.trainByInstanceGradientDescent(labels);
    } else {
      throw new IllegalArgumentException(
          String.format("Training method is not supported."));
    }
  }

  /**
   * Train by gradient descent. Get the updated weights using one training
   * instance.
   * 
   * @param trainingInstance
   * @return The weight update matrices.
   */
  private DoubleMatrix[] trainByInstanceGradientDescent(DoubleVector labels) {

    // initialize weight update matrices
    DenseDoubleMatrix[] weightUpdateMatrices = new DenseDoubleMatrix[this.weightMatrixList
        .size()];
    for (int m = 0; m < weightUpdateMatrices.length; ++m) {
      weightUpdateMatrices[m] = new DenseDoubleMatrix(this.weightMatrixList
          .get(m).getRowCount(), this.weightMatrixList.get(m).getColumnCount());
    }
    DoubleVector deltaVec = new DenseDoubleVector(
        this.layerSizeList.get(this.layerSizeList.size() - 1));

    DoubleFunction squashingFunction = this.squashingFunctionList
        .get(this.squashingFunctionList.size() - 1);

    DoubleMatrix lastWeightMatrix = this.weightMatrixList
        .get(this.weightMatrixList.size() - 1);

    for (int i = 0; i < deltaVec.getDimension(); ++i) {
      double finalOut = neurons.get(finalLayerIdx)[i].getOutput();
      double costFuncDerivative = this.costFunction.applyDerivative(
          labels.get(i), finalOut);
      // add regularization
      costFuncDerivative += this.regularizationWeight
          * lastWeightMatrix.getRowVector(i).sum();

      if (!squashingFunction.getFunctionName().equalsIgnoreCase(
          SoftMax.class.getSimpleName())) {
        costFuncDerivative *= squashingFunction.applyDerivative(finalOut);
      }

      neurons.get(finalLayerIdx)[i].setDelta(costFuncDerivative);
      deltaVec.set(i, costFuncDerivative);
    }

    // start from previous layer of output layer
    for (int layer = this.layerSizeList.size() - 2; layer >= 0; --layer) {
      backpropagate(layer, weightUpdateMatrices[layer]);
    }

    this.setPrevWeightMatrices(weightUpdateMatrices);
    return weightUpdateMatrices;
  }

  /**
   * Back-propagate the errors to from next layer to current layer. The weight
   * updated information will be stored in the weightUpdateMatrices, and the
   * delta of the prevLayer would be returned.
   * 
   * @param layer Index of current layer.
   */
  private void backpropagate(int curLayerIdx,
  // DoubleVector nextLayerDelta, DoubleVector curLayerOutput,
      DenseDoubleMatrix weightUpdateMatrix) {

    // get layer related information
    DoubleMatrix weightMatrix = this.weightMatrixList.get(curLayerIdx);
    DoubleMatrix prevWeightMatrix = this.prevWeightUpdatesList.get(curLayerIdx);

    DoubleVector deltaVector = new DenseDoubleVector(
        weightMatrix.getColumnCount());

    for (int row = 0; row < weightMatrix.getColumnCount(); ++row) {
      Neuron n = neurons.get(curLayerIdx)[row];
      n.setWeightVector(weightMatrix.getRowCount());

      List<Synapse<DoubleWritable, DoubleWritable>> msgs = new ArrayList<Synapse<DoubleWritable, DoubleWritable>>();

      for (int col = 0; col < weightMatrix.getRowCount(); ++col) {
        double deltaOfNextLayer;
        if (curLayerIdx + 1 == this.finalLayerIdx)
          deltaOfNextLayer = neurons.get(curLayerIdx + 1)[col].getDelta();
        else
          deltaOfNextLayer = neurons.get(curLayerIdx + 1)[col + 1].getDelta();

        msgs.add(new Synapse<DoubleWritable, DoubleWritable>(
            new DoubleWritable(deltaOfNextLayer), new DoubleWritable(
                weightMatrix.get(col, row)), new DoubleWritable(
                prevWeightMatrix.get(col, row))));
      }

      Iterable<Synapse<DoubleWritable, DoubleWritable>> iterable = msgs;
      try {
        n.backward(msgs);
      } catch (IOException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }

      // update weights
      weightUpdateMatrix.setColumn(row, n.getWeights());
      deltaVector.set(row, n.getDelta());
    }

  }

  @Override
  protected BSPJob trainInternal(HamaConfiguration conf) throws IOException,
      InterruptedException, ClassNotFoundException {
    this.conf = conf;
    this.fs = FileSystem.get(conf);

    String modelPath = conf.get("model.path");
    if (modelPath != null) {
      this.modelPath = modelPath;
    }
    // modelPath must be set before training
    if (this.modelPath == null) {
      throw new IllegalArgumentException(
          "Please specify the modelPath for model, "
              + "either through setModelPath() or add 'modelPath' to the training parameters.");
    }
    this.writeModelToFile();

    // create job
    BSPJob job = new BSPJob(conf, LayeredNeuralNetworkTrainer.class);
    job.setJobName("Neural Network training");
    job.setJarByClass(LayeredNeuralNetworkTrainer.class);
    job.setBspClass(LayeredNeuralNetworkTrainer.class);

    job.getConfiguration().setInt(Constants.ADDITIONAL_BSP_TASKS, 1);
    
    job.setInputPath(new Path(conf.get("training.input.path")));
    job.setInputFormat(org.apache.hama.bsp.SequenceFileInputFormat.class);
    job.setInputKeyClass(LongWritable.class);
    job.setInputValueClass(VectorWritable.class);
    job.setOutputKeyClass(NullWritable.class);
    job.setOutputValueClass(NullWritable.class);
    job.setOutputFormat(org.apache.hama.bsp.NullOutputFormat.class);

    return job;
  }

  @Override
  protected void calculateTrainingError(DoubleVector labels, DoubleVector output) {
    DoubleVector errors = labels.deepCopy().applyToElements(output,
        this.costFunction);
    this.trainingError = errors.sum();
  }

  /**
   * Get the squashing function of a specified layer.
   * 
   * @param idx
   * @return a new vector with the result of the operation.
   */
  public DoubleFunction getSquashingFunction(int idx) {
    return this.squashingFunctionList.get(idx);
  }

}
