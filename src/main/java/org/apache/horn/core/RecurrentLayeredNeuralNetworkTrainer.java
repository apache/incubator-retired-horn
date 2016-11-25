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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.BSP;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.commons.io.FloatVectorWritable;
import org.apache.hama.commons.math.DenseFloatMatrix;
import org.apache.hama.commons.math.FloatMatrix;
import org.apache.hama.commons.math.FloatVector;

/**
 * The trainer that train the {@link RecurrentLayeredNeuralNetwork} based on BSP
 * framework.
 * 
 */
public final class RecurrentLayeredNeuralNetworkTrainer
    extends
    BSP<LongWritable, FloatVectorWritable, NullWritable, NullWritable, ParameterMessage> {

  private static final Log LOG = LogFactory
      .getLog(RecurrentLayeredNeuralNetworkTrainer.class);

  private RecurrentLayeredNeuralNetwork inMemoryModel;
  private HamaConfiguration conf;
  /* Default batch size */
  private int batchSize;

  /* check the interval between intervals */
  private double prevAvgTrainingError;
  private double curAvgTrainingError;
  private long convergenceCheckInterval;
  private long iterations;
  private long maxIterations;
  private boolean isConverge;

  private String modelPath;

  @Override
  /**
   * If the model path is specified, load the existing from storage location.
   */
  public void setup(
      BSPPeer<LongWritable, FloatVectorWritable, NullWritable, NullWritable, ParameterMessage> peer) {
    if (peer.getPeerIndex() == 0) {
      LOG.info("Begin to train");
    }
    this.isConverge = false;
    this.conf = peer.getConfiguration();
    this.iterations = 0;
    this.modelPath = conf.get("model.path");
    this.maxIterations = conf.getLong("training.max.iterations", Long.MAX_VALUE);
    this.convergenceCheckInterval = conf.getLong("convergence.check.interval",
        100);
    this.inMemoryModel = new RecurrentLayeredNeuralNetwork(conf, modelPath, true);
    this.prevAvgTrainingError = Integer.MAX_VALUE;
    this.batchSize = conf.getInt("training.batch.size", 5);
  }

  @Override
  /**
   * Write the trained model back to stored location.
   */
  public void cleanup(
      BSPPeer<LongWritable, FloatVectorWritable, NullWritable, NullWritable, ParameterMessage> peer) {
    // write model to modelPath
    if (peer.getPeerIndex() == peer.getNumPeers() - 1) {
      try {
        LOG.info(String.format("End of training, number of iterations: %d.",
            this.iterations));
        LOG.info(String.format("Write model back to %s",
            inMemoryModel.getModelPath()));
        this.inMemoryModel.writeModelToFile();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  private List<FloatVector> trainingSet = new ArrayList<FloatVector>();
  private Random r = new Random();

  @Override
  public void bsp(
      BSPPeer<LongWritable, FloatVectorWritable, NullWritable, NullWritable, ParameterMessage> peer)
      throws IOException, SyncException, InterruptedException {
    // load local data into memory
    LongWritable key = new LongWritable();
    FloatVectorWritable value = new FloatVectorWritable();
    while (peer.readNext(key, value)) {
      FloatVector v = value.getVector();
      trainingSet.add(v);
    }

    if (peer.getPeerIndex() != peer.getNumPeers() - 1) {
      LOG.debug(peer.getPeerName() + ": " + trainingSet.size() + " training instances loaded.");
    }
    
    while (this.iterations++ < maxIterations) {
      this.inMemoryModel.setIterationNumber(iterations);
      
      // each groom calculate the matrices updates according to local data
      if (peer.getPeerIndex() != peer.getNumPeers() - 1) {
        calculateUpdates(peer);
      } else {
        // doing summation received updates
        if (peer.getSuperstepCount() > 0) {
          // and broadcasts previous updated weights
          mergeUpdates(peer);
        }
      }

      peer.sync();

      if (maxIterations == Long.MAX_VALUE && isConverge) {
        if (peer.getPeerIndex() == peer.getNumPeers() - 1)
          peer.sync();
        break;
      }
    }

    peer.sync();
    if (peer.getPeerIndex() == peer.getNumPeers() - 1)
      mergeUpdates(peer); // merge last updates
  }

  private FloatVector getRandomInstance() {
    return trainingSet.get(r.nextInt(trainingSet.size()));
  }

  /**
   * Calculate the matrices updates according to local partition of data.
   * 
   * @param peer
   * @throws IOException
   */
  private void calculateUpdates(
      BSPPeer<LongWritable, FloatVectorWritable, NullWritable, NullWritable, ParameterMessage> peer)
      throws IOException {
    // receive update information from master
    if (peer.getNumCurrentMessages() != 0) {
      ParameterMessage inMessage = peer.getCurrentMessage();
      FloatMatrix[] newWeights = inMessage.getCurMatrices();
      FloatMatrix[] preWeightUpdates = inMessage.getPrevMatrices();
      this.inMemoryModel.setWeightMatrices(newWeights);
      this.inMemoryModel.setPrevWeightMatrices(preWeightUpdates);
      this.isConverge = inMessage.isConverge();
      // check converge
      if (isConverge) {
        return;
      }
    }

    FloatMatrix[] weightUpdates = new FloatMatrix[this.inMemoryModel.getSizeOfWeightmatrix()];
    int matrixIdx = 0;
    for (List<FloatMatrix> aWeightMatrixList: this.inMemoryModel.weightMatrixLists) {
      for (FloatMatrix aWeightMatrix : aWeightMatrixList) {
        weightUpdates[matrixIdx++] = new DenseFloatMatrix(
            aWeightMatrix.getRowCount(), aWeightMatrix.getColumnCount());
      }
    }

    // continue to train
    float avgTrainingError = 0.0f;
    for (int recordsRead = 0; recordsRead < batchSize; ++recordsRead) {
      FloatVector trainingInstance = getRandomInstance();
      RecurrentLayeredNeuralNetwork.matricesAdd(
          weightUpdates, this.inMemoryModel.trainByInstance(trainingInstance));
      avgTrainingError += this.inMemoryModel.trainingError;
    }
    avgTrainingError /= batchSize;

    // calculate the average of updates
    for (int i = 0; i < weightUpdates.length; ++i) {
      weightUpdates[i] = weightUpdates[i].divide(batchSize);
    }

    FloatMatrix[] prevWeightUpdates = this.inMemoryModel
        .getPrevMatricesUpdates();
    ParameterMessage outMessage = new ParameterMessage(avgTrainingError, false,
        weightUpdates, prevWeightUpdates);

    peer.send(peer.getPeerName(peer.getNumPeers() - 1), outMessage);
  }

  /**
   * Merge the updates according to the updates of the grooms.
   * 
   * @param peer
   * @throws IOException
   */
  private void mergeUpdates(
      BSPPeer<LongWritable, FloatVectorWritable, NullWritable, NullWritable, ParameterMessage> peer)
      throws IOException {
    int numMessages = peer.getNumCurrentMessages();
    boolean converge = false;
    if (numMessages == 0) { // converges
      converge = true;
      return;
    }

    double avgTrainingError = 0;
    FloatMatrix[] matricesUpdates = null;
    FloatMatrix[] prevMatricesUpdates = null;

    while (peer.getNumCurrentMessages() > 0) {
      ParameterMessage message = peer.getCurrentMessage();
      if (matricesUpdates == null) {
        matricesUpdates = message.getCurMatrices();
        prevMatricesUpdates = message.getPrevMatrices();
      } else {
        RecurrentLayeredNeuralNetwork.matricesAdd(matricesUpdates,
            message.getCurMatrices());
        RecurrentLayeredNeuralNetwork.matricesAdd(prevMatricesUpdates,
            message.getPrevMatrices());
      }

      avgTrainingError += message.getTrainingError();
    }

    if (numMessages > 1) {
      avgTrainingError /= numMessages;
      for (int i = 0; i < matricesUpdates.length; ++i) {
        matricesUpdates[i] = matricesUpdates[i].divide(numMessages);
        prevMatricesUpdates[i] = prevMatricesUpdates[i].divide(numMessages);
      }
    }

    this.inMemoryModel.updateWeightMatrices(matricesUpdates);
    this.inMemoryModel.setPrevWeightMatrices(prevMatricesUpdates);

    // check convergence
    if (peer.getSuperstepCount() > 0
        && iterations % convergenceCheckInterval == 0) {
      if (prevAvgTrainingError < curAvgTrainingError) {
        // error cannot decrease any more
        converge = true;
      }
      // update
      prevAvgTrainingError = curAvgTrainingError;
      LOG.info("Training error: " + curAvgTrainingError + " at " + (iterations)
          + " iteration.");
      curAvgTrainingError = 0;
    }
    curAvgTrainingError += avgTrainingError / convergenceCheckInterval;
    this.isConverge = converge;

    if (iterations < maxIterations) {
      // broadcast updated weight matrices
      for (String peerName : peer.getAllPeerNames()) {
        ParameterMessage msg = new ParameterMessage(0, converge,
            this.inMemoryModel.getWeightMatrices(),
            this.inMemoryModel.getPrevMatricesUpdates());
        if (!peer.getPeerName().equals(peerName))
          peer.send(peerName, msg);
      }
    }
  }

}
