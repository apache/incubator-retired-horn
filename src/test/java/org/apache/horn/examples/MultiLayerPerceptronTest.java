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
package org.apache.horn.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hama.Constants;
import org.apache.hama.HamaCluster;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.commons.io.FloatVectorWritable;
import org.apache.hama.commons.math.DenseFloatVector;
import org.apache.hama.commons.math.FloatVector;
import org.apache.horn.core.Constants.TrainingMethod;
import org.apache.horn.core.HornJob;
import org.apache.horn.core.LayeredNeuralNetwork;
import org.apache.horn.examples.MultiLayerPerceptron.StandardNeuron;
import org.apache.horn.funcs.CrossEntropy;
import org.apache.horn.funcs.Sigmoid;

/**
 * Test the functionality of NeuralNetwork Example.
 */
public class MultiLayerPerceptronTest extends HamaCluster {
  private static final Log LOG = LogFactory
      .getLog(MultiLayerPerceptronTest.class);
  private HamaConfiguration conf;
  private FileSystem fs;
  private String MODEL_PATH = "/tmp/neuralnets.model";
  private String RESULT_PATH = "/tmp/neuralnets.txt";
  private String SEQTRAIN_DATA = "/tmp/test-neuralnets.data";

  public MultiLayerPerceptronTest() {
    conf = new HamaConfiguration();
    conf.set("bsp.master.address", "localhost");
    conf.setBoolean("hama.child.redirect.log.console", true);
    conf.setBoolean("hama.messenger.runtime.compression", false);
    assertEquals("Make sure master addr is set to localhost:", "localhost",
        conf.get("bsp.master.address"));
    conf.set("bsp.local.dir", "/tmp/hama-test");
    conf.set(Constants.ZOOKEEPER_QUORUM, "localhost");
    conf.setInt(Constants.ZOOKEEPER_CLIENT_PORT, 21810);
    conf.set("hama.sync.client.class",
        org.apache.hama.bsp.sync.ZooKeeperSyncClientImpl.class
            .getCanonicalName());
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    fs = FileSystem.get(conf);
  }

  @Override
  public void tearDown() throws Exception {
    super.tearDown();
  }

  public void testNeuralnetsLabeling() throws IOException {
    this.neuralNetworkTraining();

    String featureDataPath = "src/test/resources/neuralnets_classification_test.txt";
    try {
      LayeredNeuralNetwork ann = new LayeredNeuralNetwork(conf, MODEL_PATH);

      // process data in streaming approach
      FileSystem fs = FileSystem.get(new URI(featureDataPath), conf);
      BufferedReader br = new BufferedReader(new InputStreamReader(
          fs.open(new Path(featureDataPath))));

      String line = null;
      line = null;

      // compare results with ground-truth
      BufferedReader groundTruthReader = new BufferedReader(new FileReader(
          "src/test/resources/neuralnets_classification_label.txt"));

      double correct = 0;
      int samples = 0;
      while ((line = br.readLine()) != null) {
        if (line.trim().length() == 0) {
          continue;
        }
        String[] tokens = line.trim().split(",");
        float[] vals = new float[tokens.length];
        for (int i = 0; i < tokens.length; ++i) {
          vals[i] = Float.parseFloat(tokens[i]);
        }
        FloatVector instance = new DenseFloatVector(vals);
        FloatVector result = ann.getOutput(instance);
        double actual = result.toArray()[0];
        double expected = Double.parseDouble(groundTruthReader.readLine());

        LOG.info("evaluated: " + actual + ", expected: " + expected);
        if (actual < 0.5 && expected < 0.5 || actual >= 0.5 && expected >= 0.5) {
          ++correct;
        }
        samples++;
      }

      groundTruthReader.close();
      br.close();

      LOG.info("## Precision: " + (correct / samples));
      assertTrue((correct / samples) > 0.5);

    } catch (Exception e) {
      e.printStackTrace();
    } finally {
      fs.delete(new Path(RESULT_PATH), true);
      fs.delete(new Path(MODEL_PATH), true);
      fs.delete(new Path(SEQTRAIN_DATA), true);
    }
  }

  @SuppressWarnings("deprecation")
  private void neuralNetworkTraining() {
    String strTrainingDataPath = "src/test/resources/neuralnets_classification_training.txt";
    int featureDimension = 8;
    int labelDimension = 1;

    Path sequenceTrainingDataPath = new Path(SEQTRAIN_DATA);
    try {
      SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf,
          sequenceTrainingDataPath, LongWritable.class, FloatVectorWritable.class);
      BufferedReader br = new BufferedReader(
          new FileReader(strTrainingDataPath));
      String line = null;
      // convert the data in sequence file format
      while ((line = br.readLine()) != null) {
        String[] tokens = line.split(",");
        float[] vals = new float[tokens.length];
        for (int i = 0; i < tokens.length; ++i) {
          vals[i] = Float.parseFloat(tokens[i]);
        }
        writer.append(new LongWritable(), new FloatVectorWritable(
            new DenseFloatVector(vals)));
      }
      writer.close();
      br.close();
    } catch (IOException e1) {
      e1.printStackTrace();
    }

    try {
      HornJob job = new HornJob(conf, MultiLayerPerceptronTest.class);
      job.setTrainingSetPath(SEQTRAIN_DATA);
      job.setModelPath(MODEL_PATH);

      job.setMaxIteration(1000);
      job.setLearningRate(0.4f);
      job.setMomentumWeight(0.2f);
      job.setRegularizationWeight(0.001f);

      job.setConvergenceCheckInterval(100);
      job.setBatchSize(300);

      job.setTrainingMethod(TrainingMethod.GRADIENT_DESCENT);

      job.inputLayer(featureDimension, Sigmoid.class, StandardNeuron.class);
      job.addLayer(featureDimension, Sigmoid.class, StandardNeuron.class);
      job.outputLayer(labelDimension, Sigmoid.class, StandardNeuron.class);

      job.setCostFunction(CrossEntropy.class);
      
      long startTime = System.currentTimeMillis();
      if (job.waitForCompletion(true)) {
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
            / 1000.0 + " seconds");
      }

    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
