package org.apache.horn.distbelief;

import java.io.IOException;

import org.apache.hama.bsp.BSP;
import org.apache.hama.bsp.BSPPeer;
import org.apache.hama.bsp.sync.SyncException;

/**
 * This DistBeliefModelTrainer performs each SGD. 
 */
public class DistBeliefModelTrainer extends BSP {

  private boolean isConverge = false;
  private int iterations;
  private int maxIterations;
  
  @Override
  public final void setup(BSPPeer peer) {
    // loads subset of neural network model replica into memory
  }
  
  @Override
  public void bsp(BSPPeer peer) throws IOException, SyncException,
      InterruptedException {

    // Iterate until reach max iteration or convergence
    while (this.iterations++ < maxIterations) {
      
      // Fetch latest parameters
      fetchParameters(peer);
      
      // Perform mini-batch
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
   * @param peer
   */
  private void doMinibatch(BSPPeer peer) {
    double avgTrainingError = 0.0;
    // 1. loads a set of mini-batch instances from assigned splits into memory
    
    // 2. train incrementally from a mini-batch of instances
    /*
    for (Instance trainingInstance : MiniBatchSet) {
      
      // 2.1 upward propagation (start from the input layer)
      for (Neuron neuron : neurons) {  
        neuron.upward(msg);
        sync();
      }
        
      // calculate total error
      sync();
      
      // 2.3 downward propagation (start from the total error)
      for (Neuron neuron : neurons) {  
        neuron.downward(msg);
        sync();
      }
    
      // calculate the the average training error
    }
    */
    
  }
  
  private void fetchParameters(BSPPeer peer) {
    // TODO fetch latest weights from the parameter server
  }

  private void pushParameters(BSPPeer peer) {
    // TODO push updated weights     
  }

}
