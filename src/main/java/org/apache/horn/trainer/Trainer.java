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
package org.apache.horn.trainer;

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hama.bsp.BSP;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;

/**
 * The forward and backward passes are the essential computations of a Neural
 * Net. So, only few vertices of single layer of Neural Net will be activated in
 * a single superstep. This is quite inefficient. So, instead of doing like
 * this, we send training instance continuously at every superstep, and then
 * handle the information (forward messages of current training instance) and
 * error (backward messages of previous training instance) at once.
 * 
 * Then, we push the accumulated updates to parameter servers in the
 * corresponding mini-batch interval.
 * 
 */
public class Trainer extends BSP {
  
  private static final Log LOG = LogFactory.getLog(Trainer.class);
  
  private boolean isConverge = false;
  private int iterations;
  private int maxIterations;
  private int batchSize;

  @Override
  public final void setup(BSPPeer peer) {
    this.iterations = 0;
    this.maxIterations = peer.getConfiguration()
        .getInt("horn.max.iteration", 1);
    this.batchSize = peer.getConfiguration()
        .getInt("horn.minibatch.size", 1000);

    LOG.info("max iteration: " + this.maxIterations);
    
    // loads subset of neural network model replica into memory
  }

  @Override
  public void bsp(BSPPeer peer) throws IOException, SyncException,
      InterruptedException {

    // Iterate until reach max iteration or convergence
    while (this.iterations++ < maxIterations) {

      // Fetch latest parameters
      fetchParameters(peer);
      // Perform the batch
      doMinibatch(peer);
      // Push parameters
      pushParameters(peer);

      if (this.isConverge) {
        break;
      }
    }

  }

  /**
   * Performs the mini-batch
   * 
   * @param peer
   * @throws IOException 
   * @throws InterruptedException 
   * @throws SyncException 
   */
  private void doMinibatch(BSPPeer peer) throws IOException, SyncException, InterruptedException {
    double avgTrainingError = 0.0;

    int trains = 0;
    while (trains < batchSize) {
      // TODO reads and sends a single instance to first input layer
      LongWritable key = new LongWritable();
      Text value = new Text();
      
      if (!peer.readNext(key, value)) {
        peer.reopenInput();
        peer.readNext(key, value);
      }
      LOG.info(key + ", " + value);
      
      // TODO calls upward and downward methods

      peer.sync();
      trains++;
    }
  }

  private void fetchParameters(BSPPeer peer) {
    // TODO fetch latest weights from the parameter server
  }

  private void pushParameters(BSPPeer peer) {
    // TODO push updated weights
  }

}
