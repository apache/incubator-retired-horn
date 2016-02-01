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

import com.google.common.base.Preconditions;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hama.bsp.BSP;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;
import org.apache.hama.commons.io.VectorWritable;
import org.apache.hama.commons.math.DenseDoubleMatrix;
import org.apache.hama.commons.math.DoubleMatrix;
import org.apache.hama.commons.math.DoubleVector;
import org.apache.hama.ipc.RPC;
import org.mortbay.log.Log;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The trainer that train the {@link SmallLayeredNeuralNetwork} based on BSP
 * framework.
 * 
 */
public final class SmallLayeredNeuralNetworkTrainer
    extends
    BSP<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> {
  /* When given peer is master worker: base of parameter merge */
  /* When given peer is slave worker: neural network for training */
  private SmallLayeredNeuralNetwork inMemoryModel;

  /* Job configuration */
  private Configuration conf;

  /* Default batch size */
  private int batchSize;

  /* whether it is converging or not */
  private AtomicBoolean isConverge;

  /* When given peer is master worker: Asynchronous parameter merger */
  /* When given peer is slave worker: null */
  private RPC.Server merger;

  /* When given peer is master worker: null */
  /* When given peer is slave worker: proxy to Asynchronous parameter merger */
  private ParameterMerger proxy;

  /**
   * Returns true if this worker is master worker.
   *
   * @param peer
   * */
  private boolean isMaster(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer) {
    return peer.getPeerIndex() == 0;
  }

  @Override
  /**
   * If the model path is specified, load the existing from storage location.
   */
  public void setup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer) {
    // At least one master & slave worker exist.
    Preconditions.checkArgument(peer.getNumPeers() >= 2);

    String modelPath = conf.get("modelPath");
    this.inMemoryModel = new SmallLayeredNeuralNetwork(modelPath);
    this.conf = peer.getConfiguration();
    this.batchSize = conf.getInt("training.batch.size", 50);
    this.isConverge = new AtomicBoolean(false);

    int slaveCount = peer.getNumPeers() - 1;
    int mergeLimit = conf.getInt("training.max.iterations", 100000);
    int convergenceCheckInterval = peer.getNumPeers() * conf.getInt("convergence.check.interval",
        2000);
    String master = peer.getPeerName();
    String masterAddr = master.substring(0, master.indexOf(':'));
    int port = conf.getInt("sync.server.port", 40042);

    if (isMaster(peer)) {
      try {
        this.merger = RPC.getServer(new ParameterMergerServer(inMemoryModel, isConverge, slaveCount,
            mergeLimit, convergenceCheckInterval), masterAddr, port, conf);
        merger.start();
      } catch (IOException e) {
        e.printStackTrace();
      }
      Log.info("Begin to train");
    } else {
      InetSocketAddress addr = new InetSocketAddress(masterAddr, port);
      try {
        this.proxy = (ParameterMerger) RPC.getProxy(ParameterMerger.class, ParameterMerger.versionID, addr, conf);
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  @Override
  /**
   * Write the trained model back to stored location.
   */
  public void cleanup(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer) {
    // write model to modelPath
    if (isMaster(peer)) {
      try {
        Log.info(String.format("Write model back to %s\n",
            inMemoryModel.getModelPath()));
        this.inMemoryModel.writeModelToFile();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }
  }

  @Override
  public void bsp(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException, SyncException, InterruptedException {
    if (!isMaster(peer)) {
      while (!this.isConverge.get()) {
        // each slave-worker calculate the matrices updates according to local data
        // and merge them with master
        calculateUpdates(peer);
      }
    }
  }

  /**
   * Calculate the matrices updates according to local partition of data.
   * 
   * @param peer
   * @throws IOException
   */
  private void calculateUpdates(
      BSPPeer<LongWritable, VectorWritable, NullWritable, NullWritable, SmallLayeredNeuralNetworkMessage> peer)
      throws IOException {
    DoubleMatrix[] weightUpdates = new DoubleMatrix[this.inMemoryModel.weightMatrixList
        .size()];
    for (int i = 0; i < weightUpdates.length; ++i) {
      int row = this.inMemoryModel.weightMatrixList.get(i).getRowCount();
      int col = this.inMemoryModel.weightMatrixList.get(i).getColumnCount();
      weightUpdates[i] = new DenseDoubleMatrix(row, col);
    }

    // continue to train
    double avgTrainingError = 0.0;
    LongWritable key = new LongWritable();
    VectorWritable value = new VectorWritable();
    for (int recordsRead = 0; recordsRead < batchSize; ++recordsRead) {
      if (!peer.readNext(key, value)) {
        peer.reopenInput();
        peer.readNext(key, value);
      }
      DoubleVector trainingInstance = value.getVector();
      SmallLayeredNeuralNetwork.matricesAdd(weightUpdates,
          this.inMemoryModel.trainByInstance(trainingInstance));
      avgTrainingError += this.inMemoryModel.trainingError;
    }
    avgTrainingError /= batchSize;

    // calculate the average of updates
    for (int i = 0; i < weightUpdates.length; ++i) {
      weightUpdates[i] = weightUpdates[i].divide(batchSize);
    }

    // exchange parameter update with master
    SmallLayeredNeuralNetworkMessage inMessage = proxy.merge(avgTrainingError, weightUpdates,
        this.inMemoryModel.getWeightMatrices());
    DoubleMatrix[] newWeights = inMessage.getCurMatrices();
    DoubleMatrix[] preWeightUpdates = inMessage.getPrevMatrices();
    this.inMemoryModel.setWeightMatrices(newWeights);
    this.inMemoryModel.setPrevWeightMatrices(preWeightUpdates);
    this.isConverge.set(inMessage.isConverge());
  }

}
