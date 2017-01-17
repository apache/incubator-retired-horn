package org.apache.horn.examples;

import java.io.IOException;

import junit.framework.TestCase;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hama.HamaConfiguration;
import org.apache.horn.core.HornJob;
import org.apache.horn.utils.MNISTEvaluator;

public class MNISTTest extends TestCase {
  private static final Log LOG = LogFactory.getLog(MNISTTest.class);

  public void testNeuralnetsLabeling() throws IOException {
    this.neuralNetworkTraining();
    MNISTEvaluator.main(new String[] { "/tmp/model",
        "/home/edward/Downloads/t10k-images.idx3-ubyte",
        "/home/edward/Downloads/t10k-labels.idx1-ubyte" });
  }

  private void neuralNetworkTraining() {
    String strTrainingDataPath = "/home/edward/mnist.seq";
    int featureDimension = 784;
    int labelDimension = 10;

    try {
      HornJob ann = MultiLayerPerceptron.createJob(new HamaConfiguration(),
          "/tmp/model", strTrainingDataPath, 0.2f, 0.98f, 0.01f,
          featureDimension, 500, labelDimension, 10, 120);

      long startTime = System.currentTimeMillis();
      if (ann.waitForCompletion(true)) {
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
            / 1000.0 + " seconds");
      }

    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
