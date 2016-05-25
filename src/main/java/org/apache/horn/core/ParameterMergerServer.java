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
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class ParameterMergerServer implements ParameterMerger {

  private static final Log LOG = LogFactory.getLog(ParameterMergerServer.class);

  /* The parameter merge base. */
  protected LayeredNeuralNetwork inMemoryModel;

  /* To terminate or not to terminate. */
  protected AtomicBoolean isConverge;

  /* The number of slave works that request commits. */
  protected int SlaveCount;

  /* After mergeLimit, terminate whether the result is converging or not. */
  protected int mergeLimit;

  /*
   * last n training errors. converging is decided based on the average value of
   * these errors.
   */
  protected double[] trainingErrors;

  /*
   * If the average of last n training errors is smaller than this value, it is
   * converging.
   */
  protected double prevAvgTrainingError = Double.MAX_VALUE;

  /* current index for trainingErrors. */
  protected int curTrainingError = 0;

  /* how many merges have been conducted? */
  protected int mergeCount = 0;

  public ParameterMergerServer(LayeredNeuralNetwork inMemoryModel,
      AtomicBoolean isConverge, int slaveCount, int mergeLimit,
      int convergenceCheckInterval) {
    this.inMemoryModel = inMemoryModel;
    this.isConverge = isConverge;
    this.SlaveCount = slaveCount;
    this.mergeLimit = mergeLimit;
    this.trainingErrors = new double[convergenceCheckInterval];
  }

  @Override
  public long getProtocolVersion(String s, long l) throws IOException {
    return ParameterMerger.versionID;
  }

  @Override
  public ParameterMessage merge(ParameterMessage msg) {
/*
    double trainingError = msg.getTrainingError();
    DoubleMatrix[] weightUpdates = msg.getCurMatrices();
    DoubleMatrix[] prevWeightUpdates = msg.getPrevMatrices();

    Preconditions
        .checkArgument(weightUpdates.length == prevWeightUpdates.length);

    LOG.info("Start merging: " + this.mergeCount);

    if (!this.isConverge.get()) {
      synchronized (inMemoryModel) {

        LOG.info(">>>> before: " + this.inMemoryModel.getWeightMatrices()[0].get(0, 0));
        
        // this.inMemoryModel.addWeights(weightUpdates);
        // this.inMemoryModel.addPrevWeights(prevWeightUpdates);
        
        LOG.info(", after: " + this.inMemoryModel.getWeightMatrices()[0].get(0, 0));
        
        // add trainingError to trainingErrors
        this.trainingErrors[this.curTrainingError++] = trainingError;

        // check convergence
        if (this.trainingErrors.length == this.curTrainingError) {
          double curAvgTrainingError = 0.0;
          for (int i = 0; i < this.curTrainingError; ++i) {
            curAvgTrainingError += this.trainingErrors[i];
          }
          curAvgTrainingError /= this.trainingErrors.length;

          if (prevAvgTrainingError < curAvgTrainingError) {
            this.isConverge.set(true);
          } else {
            // update
            prevAvgTrainingError = curAvgTrainingError;
            this.curTrainingError = 0;
          }
        }

        if (++this.mergeCount == this.mergeLimit) {
          this.isConverge.set(true);
        }
      }
    }

    return new ParameterMessage(0, this.isConverge.get(),
        this.inMemoryModel.getWeightMatrices(),
        this.inMemoryModel.getPrevMatricesUpdates());
        */
    return null;
  }

}
