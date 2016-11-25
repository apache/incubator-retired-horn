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
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hama.Constants;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.BSPJob;
import org.apache.hama.commons.io.FloatMatrixWritable;
import org.apache.hama.commons.io.VectorWritable;
import org.apache.hama.commons.math.DenseFloatMatrix;
import org.apache.hama.commons.math.DenseFloatVector;
import org.apache.hama.commons.math.FloatFunction;
import org.apache.hama.commons.math.FloatMatrix;
import org.apache.hama.commons.math.FloatVector;
import org.apache.hama.util.ReflectionUtils;
import org.apache.horn.core.Constants.LearningStyle;
import org.apache.horn.core.Constants.TrainingMethod;
import org.apache.horn.examples.MultiLayerPerceptron.StandardNeuron;
import org.apache.horn.funcs.FunctionFactory;
import org.apache.horn.funcs.IdentityFunction;
import org.apache.horn.funcs.SoftMax;
import org.apache.horn.utils.MathUtils;

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
public class RecurrentLayeredNeuralNetwork extends AbstractLayeredNeuralNetwork {

  private static final Log LOG = LogFactory.getLog(RecurrentLayeredNeuralNetwork.class);

  /* Weights between neurons at adjacent layers */
  protected List<FloatMatrix> weightMatrixList;
  /* Weights between neurons at adjacent layers */
  protected List<List<FloatMatrix>> weightMatrixLists;
  /* Previous weight updates between neurons at adjacent layers */
  protected List<FloatMatrix> prevWeightUpdatesList;
  protected List<List<FloatMatrix>> prevWeightUpdatesLists;
  /* Different layers can have different squashing function */
  protected List<FloatFunction> squashingFunctionList;
  protected List<Class<? extends Neuron<?>>> neuronClassList;
  /* Record the recurrent layer */
  protected List<Boolean> recurrentLayerList;
  /* Recurrent step size */
  protected int recurrentStepSize;
  protected int finalLayerIdx;
  private List<Neuron<?>[]> neurons;
  private List<List<Neuron<?>[]>> neuronLists;
  private float dropRate;
  private long iterations;

  private int numOutCells;

  public RecurrentLayeredNeuralNetwork() {
    this.layerSizeList = Lists.newArrayList();
    this.weightMatrixList = Lists.newArrayList();
    this.prevWeightUpdatesList = Lists.newArrayList();
    this.squashingFunctionList = Lists.newArrayList();
    this.neuronClassList = Lists.newArrayList();
    this.weightMatrixLists = Lists.newArrayList();
    this.prevWeightUpdatesLists = Lists.newArrayList();
    this.neuronLists = Lists.newArrayList();
    this.recurrentLayerList = Lists.newArrayList();
  }

  public RecurrentLayeredNeuralNetwork(HamaConfiguration conf, String modelPath) {
    super(conf, modelPath);
    initializeNeurons(false);
    initializeWeightMatrixLists();
  }

  public RecurrentLayeredNeuralNetwork(HamaConfiguration conf, String modelPath,
      boolean isTraining) {
    super(conf, modelPath);
    initializeNeurons(isTraining);
    initializeWeightMatrixLists();
  }

  /**
   *  initialize neuron objects
   * @param isTraining
   */
  private void initializeNeurons(boolean isTraining) {
    this.neuronLists = Lists.newArrayListWithExpectedSize(recurrentStepSize);
    for (int stepIdx = 0; stepIdx < this.recurrentStepSize; stepIdx++) {
      neurons = new ArrayList<Neuron<?>[]>();
      
      int expectedNeuronsSize = this.layerSizeList.size();
      if (stepIdx < this.recurrentStepSize - this.numOutCells) {
        expectedNeuronsSize--;
      }
      for (int neuronLayerIdx = 0; neuronLayerIdx < expectedNeuronsSize; neuronLayerIdx++) {
        int numOfNeurons = layerSizeList.get(neuronLayerIdx);
        // if not final layer and next layer is recurrent
        if (stepIdx > 0 && neuronLayerIdx < layerSizeList.size() - 1 
            &&  this.recurrentLayerList.get(neuronLayerIdx+1)) {
          numOfNeurons = numOfNeurons + layerSizeList.get(neuronLayerIdx+1) - 1;
        }
        Class<? extends Neuron<?>> neuronClass;
        if (neuronLayerIdx == 0)
          neuronClass = StandardNeuron.class; // actually doesn't needed
        else
          neuronClass = neuronClassList.get(neuronLayerIdx - 1);

        Neuron<?>[] tmp = new Neuron[numOfNeurons];
        for (int neuronIdx = 0; neuronIdx < numOfNeurons; neuronIdx++) {
          Neuron<?> n = newNeuronInstance(neuronClass);
          if (n instanceof RecurrentDropoutNeuron)
            ((RecurrentDropoutNeuron) n).setDropRate(dropRate);
          if (neuronLayerIdx > 0 && neuronIdx < layerSizeList.get(neuronLayerIdx))
            n.setSquashingFunction(squashingFunctionList.get(neuronLayerIdx - 1));
          else
            n.setSquashingFunction(new IdentityFunction());
          n.setLayerIndex(neuronLayerIdx);
          n.setNeuronID(neuronIdx);
          n.setLearningRate(this.learningRate);
          n.setMomentumWeight(this.momentumWeight);
          n.setTraining(isTraining);
          tmp[neuronIdx] = n;
        }
        neurons.add(tmp);
      }
      this.neuronLists.add(neurons);
    }
  }

  /**
   * Initialize WeightMatrixLists
   */
  public void initializeWeightMatrixLists() {
    this.numOutCells = (numOutCells == 0 ? this.recurrentStepSize:numOutCells);
    this.weightMatrixLists.clear();
    this.weightMatrixLists = Lists.newArrayListWithExpectedSize(this.recurrentStepSize);
    this.prevWeightUpdatesLists.clear();
    this.prevWeightUpdatesLists = Lists.newArrayListWithExpectedSize(this.recurrentStepSize);

    for (int stepIdx = 0; stepIdx < recurrentStepSize - 1; stepIdx++) {
      int expectedMatrixListSize = this.layerSizeList.size() - 1;
      if (stepIdx < this.recurrentStepSize - this.numOutCells) {
        expectedMatrixListSize--;
      }
      List<FloatMatrix> aWeightMatrixList = Lists.newArrayListWithExpectedSize(
          expectedMatrixListSize);
      List<FloatMatrix> aPrevWeightUpdatesList = Lists.newArrayListWithExpectedSize(
          expectedMatrixListSize);
      for (int matrixIdx = 0; matrixIdx < expectedMatrixListSize; matrixIdx++) {
        int rows = this.weightMatrixList.get(matrixIdx).getRowCount();
        int cols = this.weightMatrixList.get(matrixIdx).getColumnCount();
        if ( stepIdx == 0 )
          cols = this.layerSizeList.get(matrixIdx);
        FloatMatrix weightMatrix = new DenseFloatMatrix(rows, cols);
        weightMatrix.applyToElements(new FloatFunction() {
          @Override
          public float apply(float value) {
            return RandomUtils.nextFloat() - 0.5f;
          }
          @Override
          public float applyDerivative(float value) {
            throw new UnsupportedOperationException("");
          }
        });
        aWeightMatrixList.add(weightMatrix);
        aPrevWeightUpdatesList.add(
            new DenseFloatMatrix(
                this.prevWeightUpdatesList.get(matrixIdx).getRowCount(),
                this.prevWeightUpdatesList.get(matrixIdx).getColumnCount()));
      }
      this.weightMatrixLists.add(aWeightMatrixList);
      this.prevWeightUpdatesLists.add(aPrevWeightUpdatesList);
    }
    // add matrix of last step
    this.weightMatrixLists.add(this.weightMatrixList);
    this.prevWeightUpdatesLists.add(this.prevWeightUpdatesList);
    this.weightMatrixList = Lists.newArrayList();
    this.prevWeightUpdatesList = Lists.newArrayList();
  }

  @Override
  /**
   * {@inheritDoc}
   */
  public int addLayer(int size, boolean isFinalLayer,
      FloatFunction squashingFunction, Class<? extends Neuron<?>> neuronClass) {
    return addLayer(size, isFinalLayer, squashingFunction, neuronClass, null, true);
  }

  public int addLayer(int size, boolean isFinalLayer,
      FloatFunction squashingFunction, Class<? extends Neuron<?>> neuronClass, int numOutCells) {
    if (isFinalLayer)
      this.numOutCells = (numOutCells == 0 ? this.recurrentStepSize:numOutCells);
    return addLayer(size, isFinalLayer, squashingFunction, neuronClass, null, false);
  }

  public int addLayer(int size, boolean isFinalLayer,
      FloatFunction squashingFunction, Class<? extends Neuron<?>> neuronClass,
      Class<? extends IntermediateOutput> interlayer, boolean isRecurrent) {
    Preconditions.checkArgument(size > 0,
        "Size of layer must be larger than 0.");
    if (!isFinalLayer) {
      if (this.layerSizeList.size() == 0) {
        this.recurrentLayerList.add(false);
        LOG.info("add input layer: " + size + " neurons");
      } else {
        this.recurrentLayerList.add(isRecurrent);
        LOG.info("add hidden layer: " + size + " neurons");
      }
      size += 1;
    } else {
      this.recurrentLayerList.add(false);
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
      // expand matrix for recurrent layer
      int col = !(this.recurrentLayerList.get(layerIdx)) ?
          sizePrevLayer : sizePrevLayer + this.layerSizeList.get(layerIdx) - 1;

      FloatMatrix weightMatrix = new DenseFloatMatrix(row, col);
      // initialize weights
      weightMatrix.applyToElements(new FloatFunction() {
        @Override
        public float apply(float value) {
          return RandomUtils.nextFloat() - 0.5f;
        }

        @Override
        public float applyDerivative(float value) {
          throw new UnsupportedOperationException("");
        }
      });
      this.weightMatrixList.add(weightMatrix);
      this.prevWeightUpdatesList.add(new DenseFloatMatrix(row, col));
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
  public void updateWeightMatrices(FloatMatrix[] matrices) {
    int matrixIdx = 0;
    for (List<FloatMatrix> aWeightMatrixList: this.weightMatrixLists) {
      for (int weightMatrixIdx = 0; weightMatrixIdx < aWeightMatrixList.size(); weightMatrixIdx++) {
        FloatMatrix matrix = aWeightMatrixList.get(weightMatrixIdx);
        aWeightMatrixList.set(weightMatrixIdx, matrix.add(matrices[matrixIdx++]));
      }
    }
  }

  /**
   * Set the previous weight matrices.
   * 
   * @param prevUpdates
   */
  void setPrevWeightMatrices(FloatMatrix[] prevUpdates) {
    int matrixIdx = 0;
    for (List<FloatMatrix> aWeightUpdateMatrixList: this.prevWeightUpdatesLists) {
      for (int weightMatrixIdx = 0; weightMatrixIdx < aWeightUpdateMatrixList.size();
          weightMatrixIdx++) {
        aWeightUpdateMatrixList.set(weightMatrixIdx, prevUpdates[matrixIdx++]);
      }
    }
  }

  /**
   * Add a batch of matrices onto the given destination matrices.
   * 
   * @param destMatrices
   * @param sourceMatrices
   */
  static void matricesAdd(FloatMatrix[] destMatrices,
      FloatMatrix[] sourceMatrices) {
    for (int i = 0; i < destMatrices.length; ++i) {
      destMatrices[i] = destMatrices[i].add(sourceMatrices[i]);
    }
  }

  /**
   * Get all the weight matrices.
   * 
   * @return The matrices in form of matrix array.
   */
  FloatMatrix[] getWeightMatrices() {
    FloatMatrix[] matrices = new FloatMatrix[this.getSizeOfWeightmatrix()];
    int matrixIdx = 0;
    for (List<FloatMatrix> aWeightMatrixList: this.weightMatrixLists) {
      for (FloatMatrix aWeightMatrix : aWeightMatrixList) {
        matrices[matrixIdx++] = aWeightMatrix;
      }
    }
    return matrices;
  }

  /**
   * Set the weight matrices.
   * 
   * @param matrices
   */
  public void setWeightMatrices(FloatMatrix[] matrices) {
    int matrixIdx = 0;
    for (List<FloatMatrix> aWeightMatrixList: this.weightMatrixLists) {
      for (int weightMatrixIdx = 0; weightMatrixIdx < aWeightMatrixList.size(); weightMatrixIdx++) {
        aWeightMatrixList.set(weightMatrixIdx, matrices[matrixIdx++]);
      }
    }
  }

  /**
   * Get the previous matrices updates in form of array.
   * 
   * @return The matrices in form of matrix array.
   */
  public FloatMatrix[] getPrevMatricesUpdates() {
    FloatMatrix[] matrices = new FloatMatrix[this.getSizeOfWeightmatrix()];
    int matrixIdx = 0;
    for (List<FloatMatrix> aWeightMatrixList: this.prevWeightUpdatesLists) {
      for (FloatMatrix aWeightMatrix : aWeightMatrixList) {
        matrices[matrixIdx++] = aWeightMatrix;
      }
    }
    return matrices;
  }

  public void setWeightMatrix(int index, FloatMatrix matrix) {
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
    this.dropRate = input.readFloat();

    // read neuron classes
    int neuronClasses = input.readInt();
    this.neuronClassList = Lists.newArrayList();
    for (int i = 0; i < neuronClasses; ++i) {
      try {
        Class<? extends Neuron<?>> clazz = (Class<? extends Neuron<?>>) Class
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
          .createFloatFunction(WritableUtils.readString(input)));
    }

    this.recurrentStepSize = input.readInt();
    this.numOutCells = input.readInt();
    int recurrentLayerListSize = input.readInt();
    this.recurrentLayerList = Lists.newArrayList();
    for (int i = 0; i < recurrentLayerListSize; i++) {
      this.recurrentLayerList.add(input.readBoolean());
    }

    // read weights and construct matrices of previous updates
    int numOfMatrices = input.readInt();
    this.weightMatrixLists = Lists.newArrayListWithExpectedSize(this.recurrentStepSize);
    this.prevWeightUpdatesLists = Lists.newArrayList();

    for (int step = 0; step < this.recurrentStepSize; step++) {
      this.weightMatrixList = Lists.newArrayList();
      this.prevWeightUpdatesList = Lists.newArrayList();

      for (int j = 0; j < this.layerSizeList.size() - 2; j++) {
        FloatMatrix matrix = FloatMatrixWritable.read(input);
        this.weightMatrixList.add(matrix);
        this.prevWeightUpdatesList.add(new DenseFloatMatrix(matrix.getRowCount(),
            matrix.getColumnCount()));
      }
      // if the cell has output layer, read from input
      if (step >= this.recurrentStepSize - this.numOutCells) {
        FloatMatrix matrix = FloatMatrixWritable.read(input);
        this.weightMatrixList.add(matrix);
        this.prevWeightUpdatesList.add(new DenseFloatMatrix(matrix.getRowCount(),
            matrix.getColumnCount()));
      }
      this.weightMatrixLists.add(this.weightMatrixList);
      this.prevWeightUpdatesLists.add(this.prevWeightUpdatesList);
    }
  }

//  }
  protected int getSizeOfWeightmatrix() {
    return this.recurrentStepSize * (this.layerSizeList.size() - 2) + this.numOutCells;
  }

  @Override
  public void write(DataOutput output) throws IOException {
    super.write(output);
    output.writeInt(finalLayerIdx);
    output.writeFloat(dropRate);

    // write neuron classes
    output.writeInt(this.neuronClassList.size());
    for (Class<? extends Neuron<?>> clazz : this.neuronClassList) {
      output.writeUTF(clazz.getName());
    }

    // write squashing functions
    output.writeInt(this.squashingFunctionList.size());
    for (FloatFunction aSquashingFunctionList : this.squashingFunctionList) {
      WritableUtils.writeString(output,
          aSquashingFunctionList.getFunctionName());
    }

    // write recurrent step size
    output.writeInt(this.recurrentStepSize);

    // write recurrent step size
    output.writeInt(this.numOutCells);

    // write recurrent layer list
    output.writeInt(this.recurrentLayerList.size());
    for (Boolean isReccurentLayer: recurrentLayerList) {
      output.writeBoolean(isReccurentLayer);
    }

    // write weight matrices
    output.writeInt(this.getSizeOfWeightmatrix());
    for (List<FloatMatrix> aWeightMatrixLists : this.weightMatrixLists) {
      for (FloatMatrix aWeightMatrixList : aWeightMatrixLists) {
        FloatMatrixWritable.write(aWeightMatrixList, output);
      }
    }

    // DO NOT WRITE WEIGHT UPDATE
  }

  @Override
  public FloatMatrix getWeightsByLayer(int layerIdx) {
    return this.weightMatrixList.get(layerIdx);
  }

  public FloatMatrix getWeightsByLayer(int stepIdx, int layerIdx) {
    return this.weightMatrixLists.get(stepIdx).get(layerIdx);
  }

  /**
   * Get the output of the model according to given feature instance.
   */
  @Override
  public FloatVector getOutput(FloatVector instance) {
    Preconditions.checkArgument((this.layerSizeList.get(0) - 1) * this.recurrentStepSize
        == instance.getDimension(), String.format(
            "The dimension of input instance should be %d.",
            this.layerSizeList.get(0) - 1));
    // transform the features to another space
    FloatVector transformedInstance = this.featureTransformer
        .transform(instance);
    // add bias feature
    FloatVector instanceWithBias = new DenseFloatVector(
        transformedInstance.getDimension() + 1);
    instanceWithBias.set(0, 0.99999f); // set bias to be a little bit less than
                                       // 1.0
    for (int i = 1; i < instanceWithBias.getDimension(); ++i) {
      instanceWithBias.set(i, transformedInstance.get(i - 1));
    }
    // return the output of the last layer
    return getOutputInternal(instanceWithBias);
  }

  public void setDropRateOfInputLayer(float dropRate) {
    this.dropRate = dropRate;
  }

  /**
   * Calculate output internally, the intermediate output of each layer will be
   * stored.
   * 
   * @param instanceWithBias The instance contains the features.
   * @return Cached output of each layer.
   */
  public FloatVector getOutputInternal(FloatVector instanceWithBias) {
    // sets the output of input layer
    Neuron<?>[] inputLayer;
    for (int stepIdx = 0; stepIdx < this.weightMatrixLists.size(); stepIdx++) {
      inputLayer = neuronLists.get(stepIdx).get(0);
      for (int inputNeuronIdx = 0; inputNeuronIdx < this.layerSizeList.get(0); inputNeuronIdx++) {
        float m2 = MathUtils.getBinomial(1, dropRate);
        if(m2 == 0)
          inputLayer[inputNeuronIdx].setDrop(true);
        else
          inputLayer[inputNeuronIdx].setDrop(false);
        inputLayer[inputNeuronIdx].setOutput(
            instanceWithBias.get(stepIdx * this.layerSizeList.get(0) + inputNeuronIdx) * m2);
      }
      // loop forward as much as recurrent step size
      this.weightMatrixList = this.weightMatrixLists.get(stepIdx);
      for (int layerIdx = 0; layerIdx < weightMatrixList.size(); ++layerIdx) {
        forward(stepIdx, layerIdx);
      }
    }

    // output for each recurrent step
    int singleOutputLength = 
        neuronLists.get(this.recurrentStepSize-1).get(this.finalLayerIdx).length;
    FloatVector output = new DenseFloatVector(singleOutputLength * this.numOutCells);
    int outputNeuronIdx = 0;
    for (int step = this.recurrentStepSize - this.numOutCells; 
        step < this.recurrentStepSize; step++) {
      neurons = neuronLists.get(step);
      for (int neuronIdx = 0; neuronIdx < singleOutputLength; neuronIdx++) {
        output.set(outputNeuronIdx, neurons.get(this.finalLayerIdx)[neuronIdx].getOutput());
        outputNeuronIdx++;
      }
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

  public class InputMessageIterable implements
      Iterable<Synapse<FloatWritable, FloatWritable>> {
    private int currNeuronID;
    private int prevNeuronID;
    private int end;
    private FloatMatrix weightMat;
    private Neuron<?>[] layer;

    public InputMessageIterable(int fromLayer, int row) {
      this.currNeuronID = row;
      this.prevNeuronID = -1;
      this.end = weightMatrixList.get(fromLayer).getColumnCount() - 1;
      this.weightMat = weightMatrixList.get(fromLayer);
      this.layer = neurons.get(fromLayer);
    }

    @Override
    public Iterator<Synapse<FloatWritable, FloatWritable>> iterator() {
      return new MessageIterator();
    }

    private class MessageIterator implements
        Iterator<Synapse<FloatWritable, FloatWritable>> {

      @Override
      public boolean hasNext() {
        if (prevNeuronID < end) {
          return true;
        } else {
          return false;
        }
      }

      private FloatWritable i = new FloatWritable();
      private FloatWritable w = new FloatWritable();
      private Synapse<FloatWritable, FloatWritable> msg = new Synapse<FloatWritable, FloatWritable>();
      
      @Override
      public Synapse<FloatWritable, FloatWritable> next() {
        prevNeuronID++;
        i.set(layer[prevNeuronID].getOutput());
        w.set(weightMat.get(currNeuronID, prevNeuronID));
        msg.set(prevNeuronID, i, w);
        return new Synapse<FloatWritable, FloatWritable>(prevNeuronID, i, w);
      }
    
      @Override
      public void remove() {
      }
    }
  }

  /**
   * Forward the calculation for one layer.
   * 
   * @param fromLayerIdx The index of the previous layer.
   */
  protected void forward(int stepIdx, int fromLayerIdx) {
    neurons = this.neuronLists.get(stepIdx);
    int curLayerIdx = fromLayerIdx + 1;
    // weight matrix for current layer
    FloatMatrix weightMatrix = this.weightMatrixList.get(fromLayerIdx);
    FloatFunction squashingFunction = getSquashingFunction(fromLayerIdx);
    FloatVector vec = new DenseFloatVector(weightMatrix.getRowCount());

    for (int row = 0; row < weightMatrix.getRowCount(); row++) {
      Neuron<?> n;
      if (curLayerIdx == finalLayerIdx)
        n = neurons.get(curLayerIdx)[row];
      else
        n = neurons.get(curLayerIdx)[row + 1];

      try {
        Iterable msgs = new InputMessageIterable(fromLayerIdx, row);
        ((RecurrentDropoutNeuron) n).setRecurrentDelta(0);
        n.setIterationNumber(iterations);
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

    // copy output to next recurrent layer
    if (this.recurrentLayerList.get(curLayerIdx) && stepIdx < this.recurrentStepSize - 1) {
      for (int i = 0; i < vec.getDimension(); i++) {
        this.neuronLists.get(stepIdx+1).get(
            fromLayerIdx)[this.layerSizeList.get(fromLayerIdx)+i].setOutput(vec.get(i));
      }
    }
  }

  /**
   * Train the model online.
   * 
   * @param trainingInstance
   */
  public void trainOnline(FloatVector trainingInstance) {
    FloatMatrix[] updateMatrices = this.trainByInstance(trainingInstance);
    this.updateWeightMatrices(updateMatrices);
  }

  @Override
  public FloatMatrix[] trainByInstance(FloatVector trainingInstance) {
    int inputDimension =  (this.layerSizeList.get(0) - 1) * this.recurrentStepSize;
    FloatVector transformedVector = this.featureTransformer.transform(
        trainingInstance.sliceUnsafe(inputDimension));
    int outputDimension;
    FloatVector inputInstance = null;
    FloatVector labels = null;
    if (this.learningStyle == LearningStyle.SUPERVISED) {
      outputDimension = this.layerSizeList.get(this.layerSizeList.size() - 1);
        // validate training instance
        Preconditions.checkArgument(
            (inputDimension + outputDimension == trainingInstance.getDimension()
            || inputDimension + outputDimension * recurrentStepSize == trainingInstance.getDimension()),
            String
                .format(
                    "The dimension of training instance is %d, but requires %d.",
                    trainingInstance.getDimension(), inputDimension + outputDimension));

      inputInstance = new DenseFloatVector(this.layerSizeList.get(0) * this.recurrentStepSize);
      // get the features from the transformed vector
      int vecIdx = 0;
      for (int i = 0; i < inputInstance.getLength(); ++i) {
        if (i % this.layerSizeList.get(0) == 0) {
          inputInstance.set(i, 1); // add bias
        } else {
          inputInstance.set(i, transformedVector.get(vecIdx));
          vecIdx++;
        }
      }
      // get the labels from the original training instance
      labels = trainingInstance.sliceUnsafe(transformedVector.getDimension(),
          trainingInstance.getDimension() - 1);
    } else if (this.learningStyle == LearningStyle.UNSUPERVISED) {
      // labels are identical to input features
      outputDimension = inputDimension;
      // validate training instance
      Preconditions.checkArgument(inputDimension == trainingInstance
          .getDimension(), String.format(
          "The dimension of training instance is %d, but requires %d.",
          trainingInstance.getDimension(), inputDimension));

      inputInstance = new DenseFloatVector(this.layerSizeList.get(0) * this.recurrentStepSize);
      // get the features from the transformed vector
      int vecIdx = 0;
      for (int i = 0; i < inputInstance.getLength(); ++i) {
        if (i % this.layerSizeList.get(0) == 0) {
          inputInstance.set(i, 1); // add bias
        } else {
          inputInstance.set(i, transformedVector.get(vecIdx));
          vecIdx++;
        }
      }
      // get the labels by copying the transformed vector
      labels = transformedVector.deepCopy();
    }
    FloatVector output = this.getOutputInternal(inputInstance);
    // get the training error
    calculateTrainingError(labels, output);
    if (this.trainingMethod.equals(TrainingMethod.GRADIENT_DESCENT)) {
      FloatMatrix[] updates = this.trainByInstanceGradientDescent(labels);
      return updates;
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
  private FloatMatrix[] trainByInstanceGradientDescent(FloatVector labels) {

    // initialize weight update matrices
    DenseFloatMatrix[] weightUpdateMatrices = new DenseFloatMatrix[this.getSizeOfWeightmatrix()];
    int matrixIdx = 0;
    for (List<FloatMatrix> aWeightMatrixList: this.weightMatrixLists) {
      for (FloatMatrix aWeightMatrix : aWeightMatrixList) {
        weightUpdateMatrices[matrixIdx++] = 
            new DenseFloatMatrix(aWeightMatrix.getRowCount(), aWeightMatrix.getColumnCount());
      }
    }
    FloatVector deltaVec = new DenseFloatVector(
        this.layerSizeList.get(layerSizeList.size() - 1) * this.numOutCells);

    FloatFunction squashingFunction = this.squashingFunctionList
        .get(this.squashingFunctionList.size() - 1);

    int labelIdx = 0;
    // start from last recurrent step to first recurrent step
    for (int step=this.recurrentStepSize-this.numOutCells; step < this.recurrentStepSize; step++) {

      FloatMatrix lastWeightMatrix = this.weightMatrixLists.get(step)
          .get(this.weightMatrixLists.get(step).size() - 1);
      int neuronIdx = 0;
      for (Neuron<?> aNeurons : this.neuronLists.get(step).get(this.finalLayerIdx)) {
        float finalOut = aNeurons.getOutput();
        float costFuncDerivative = this.costFunction.applyDerivative(
            labels.get(labelIdx), finalOut);
        // add regularization
        costFuncDerivative += this.regularizationWeight
            * lastWeightMatrix.getRowVector(neuronIdx).sum();
  
        if (!squashingFunction.getFunctionName().equalsIgnoreCase(
            SoftMax.class.getSimpleName())) {
          costFuncDerivative *= squashingFunction.applyDerivative(finalOut);
        }
        aNeurons.backpropagate(costFuncDerivative);
        deltaVec.set(labelIdx, costFuncDerivative);
        neuronIdx++;
        labelIdx++;
      }
    }

    // start from last recurrent step to first recurrent step
    boolean skipLastLayer = false;
    int weightMatrixIdx = weightUpdateMatrices.length - 1;
    for (int step = this.recurrentStepSize - 1 ; step >= 0; --step) {
      this.weightMatrixList = this.weightMatrixLists.get(step);
      this.prevWeightUpdatesList = this.prevWeightUpdatesLists.get(step);
      this.neurons = this.neuronLists.get(step);
      if (step < this.recurrentStepSize - this.numOutCells)
        skipLastLayer = true;
      // start from previous layer of output layer
      for (int layer = this.layerSizeList.size() - 2; layer >= 0; --layer) {
        if (skipLastLayer) {
          skipLastLayer = false; continue;
        }
        backpropagate(step, layer, weightUpdateMatrices[weightMatrixIdx--]);
      }
    }
    // TODO eliminate non-output cells from weightUpdateLists
    this.setPrevWeightMatrices(weightUpdateMatrices);
    return weightUpdateMatrices;
  }

  public class ErrorMessageIterable implements
      Iterable<Synapse<FloatWritable, FloatWritable>> {
    private int row;
    private int neuronID;
    private int end;
    private FloatMatrix weightMat;
    private FloatMatrix prevWeightMat;

    private float[] nextLayerDelta;
    
    public ErrorMessageIterable(int recurrentStepIdx, int curLayerIdx, int row) {
      this.row = row;
      this.neuronID = -1;
      this.weightMat = weightMatrixLists.get(recurrentStepIdx).get(curLayerIdx);
      this.end = weightMat.getRowCount() - 1;
      this.prevWeightMat = prevWeightUpdatesLists.get(recurrentStepIdx).get(curLayerIdx);
      
      Neuron<?>[] nextLayer = neuronLists.get(recurrentStepIdx).get(curLayerIdx + 1);
      nextLayerDelta = new float[weightMat.getRowCount()];
      
      for(int i = 0; i <= end; ++i) {
        if (curLayerIdx + 1 == finalLayerIdx) {
          nextLayerDelta[i] = nextLayer[i].getDelta();
        } else {
          nextLayerDelta[i] = nextLayer[i + 1].getDelta();
        }
      }
    }

    @Override
    public Iterator<Synapse<FloatWritable, FloatWritable>> iterator() {
      return new MessageIterator();
    }

    private class MessageIterator implements
        Iterator<Synapse<FloatWritable, FloatWritable>> {

      @Override
      public boolean hasNext() {
        if (neuronID < end) {
          return true;
        } else {
          return false;
        }
      }

      private FloatWritable d = new FloatWritable();
      private FloatWritable w = new FloatWritable();
      private FloatWritable p = new FloatWritable();
      private Synapse<FloatWritable, FloatWritable> msg = new Synapse<FloatWritable, FloatWritable>();
      
      @Override
      public Synapse<FloatWritable, FloatWritable> next() {
        neuronID++;
        
        d.set(nextLayerDelta[neuronID]);
        w.set(weightMat.get(neuronID, row));
        p.set(prevWeightMat.get(neuronID, row));
        msg.set(neuronID, d, w, p);
        return msg;
      }

      @Override
      public void remove() {
      }

    }
  }

  /**
   * Back-propagate the errors to from next layer to current layer. The weight
   * updated information will be stored in the weightUpdateMatrices, and the
   * delta of the prevLayer would be returned.
   * 
   * @param layer Index of current layer.
   */
  private void backpropagate(int recurrentStepIdx, int curLayerIdx,
      DenseFloatMatrix weightUpdateMatrix) {

    // get layer related information
    int x = this.weightMatrixList.get(curLayerIdx).getColumnCount();
    int y = this.weightMatrixList.get(curLayerIdx).getRowCount();

    FloatVector deltaVector = new DenseFloatVector(x);
    Neuron<?>[] ns = this.neuronLists.get(recurrentStepIdx).get(curLayerIdx);

    for (int row = 0; row < x; ++row) {
      Neuron<?> n = ns[row];
      n.setWeightVector(y);

      try {
        Iterable msgs = new ErrorMessageIterable(recurrentStepIdx, curLayerIdx, row);
        n.backward(msgs);
        if (row >= layerSizeList.get(curLayerIdx) && recurrentStepIdx > 0
            && recurrentLayerList.get(curLayerIdx+1)) {
          Neuron<?> recurrentNeuron = neuronLists.get(recurrentStepIdx-1).get(curLayerIdx+1)
              [row-layerSizeList.get(curLayerIdx)+1];
          recurrentNeuron.backpropagate(n.getDelta());
        }
      } catch (IOException e) {
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
    this.setRecurrentStepSize(conf.getInt("training.recurrent.step.size", 1));
    this.initializeWeightMatrixLists();
    this.writeModelToFile();

    // create job
    BSPJob job = new BSPJob(conf, RecurrentLayeredNeuralNetworkTrainer.class);
    job.setJobName("Neural Network training");
    job.setJarByClass(RecurrentLayeredNeuralNetworkTrainer.class);
    job.setBspClass(RecurrentLayeredNeuralNetworkTrainer.class);

    job.getConfiguration().setInt(Constants.ADDITIONAL_BSP_TASKS, 1);

    job.setBoolean("training.mode", true);
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
  protected void calculateTrainingError(FloatVector labels, FloatVector output) {
    FloatVector errors = labels.deepCopy().applyToElements(output,
        this.costFunction);
    this.trainingError = errors.sum();
  }

  /**
   * Get the squashing function of a specified layer.
   * 
   * @param idx
   * @return a new vector with the result of the operation.
   */
  public FloatFunction getSquashingFunction(int idx) {
    return this.squashingFunctionList.get(idx);
  }

  public void setIterationNumber(long iterations) {
    this.iterations = iterations;
  }

  public void setRecurrentStepSize(int recurrentStepSize) {
    this.recurrentStepSize = recurrentStepSize;
  }

}
