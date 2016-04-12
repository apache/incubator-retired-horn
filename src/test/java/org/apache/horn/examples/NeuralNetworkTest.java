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
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hama.Constants;
import org.apache.hama.HamaCluster;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.commons.io.VectorWritable;
import org.apache.hama.commons.math.DenseDoubleVector;
import org.apache.hama.commons.math.DoubleVector;
import org.apache.hama.commons.math.FunctionFactory;
import org.apache.horn.bsp.SmallLayeredNeuralNetwork;

/**
 * Test the functionality of NeuralNetwork Example.
 * 
 */
public class NeuralNetworkTest extends HamaCluster {
  private HamaConfiguration conf;
  private FileSystem fs;
  private String MODEL_PATH = "/tmp/neuralnets.model";
  private String RESULT_PATH = "/tmp/neuralnets.txt";
  private String SEQTRAIN_DATA = "/tmp/test-neuralnets.data";

  public NeuralNetworkTest() {
    conf = new HamaConfiguration();
    conf.set("bsp.master.address", "localhost");
    conf.setBoolean("hama.child.redirect.log.console", true);
    conf.setBoolean("hama.messenger.runtime.compression", true);
    assertEquals("Make sure master addr is set to localhost:", "localhost",
        conf.get("bsp.master.address"));
    conf.set("bsp.local.dir", "/tmp/hama-test");
    conf.set(Constants.ZOOKEEPER_QUORUM, "localhost");
    conf.setBoolean(Constants.FORCE_SET_BSP_TASKS, true);
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
      SmallLayeredNeuralNetwork ann = new SmallLayeredNeuralNetwork(conf,
          MODEL_PATH);

      // process data in streaming approach
      FileSystem fs = FileSystem.get(new URI(featureDataPath), conf);
      BufferedReader br = new BufferedReader(new InputStreamReader(
          fs.open(new Path(featureDataPath))));
      Path outputPath = new Path(RESULT_PATH);
      if (fs.exists(outputPath)) {
        fs.delete(outputPath, true);
      }
      BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(
          fs.create(outputPath)));

      String line = null;

      while ((line = br.readLine()) != null) {
        if (line.trim().length() == 0) {
          continue;
        }
        String[] tokens = line.trim().split(",");
        double[] vals = new double[tokens.length];
        for (int i = 0; i < tokens.length; ++i) {
          vals[i] = Double.parseDouble(tokens[i]);
        }
        DoubleVector instance = new DenseDoubleVector(vals);
        DoubleVector result = ann.getOutput(instance);
        double[] arrResult = result.toArray();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < arrResult.length; ++i) {
          sb.append(arrResult[i]);
          if (i != arrResult.length - 1) {
            sb.append(",");
          } else {
            sb.append("\n");
          }
        }
        bw.write(sb.toString());
      }

      br.close();
      bw.close();

      // compare results with ground-truth
      BufferedReader groundTruthReader = new BufferedReader(new FileReader(
          "src/test/resources/neuralnets_classification_label.txt"));
      List<Double> groundTruthList = new ArrayList<Double>();
      line = null;
      while ((line = groundTruthReader.readLine()) != null) {
        groundTruthList.add(Double.parseDouble(line));
      }
      groundTruthReader.close();

      BufferedReader resultReader = new BufferedReader(new FileReader(
          RESULT_PATH));
      List<Double> resultList = new ArrayList<Double>();
      while ((line = resultReader.readLine()) != null) {
        resultList.add(Double.parseDouble(line));
      }
      resultReader.close();
      int total = resultList.size();
      double correct = 0;
      for (int i = 0; i < groundTruthList.size(); ++i) {
        double actual = resultList.get(i);
        double expected = groundTruthList.get(i);
        LOG.info("evaluated: " + actual + ", expected: " + expected);
        if (actual < 0.5 && expected < 0.5 || actual >= 0.5 && expected >= 0.5) {
          ++correct;
        }
      }

      LOG.info("## Precision: " + (correct / total));
      assertTrue((correct / total) > 0.5);

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
          sequenceTrainingDataPath, LongWritable.class, VectorWritable.class);
      BufferedReader br = new BufferedReader(
          new FileReader(strTrainingDataPath));
      String line = null;
      // convert the data in sequence file format
      while ((line = br.readLine()) != null) {
        String[] tokens = line.split(",");
        double[] vals = new double[tokens.length];
        for (int i = 0; i < tokens.length; ++i) {
          vals[i] = Double.parseDouble(tokens[i]);
        }
        writer.append(new LongWritable(), new VectorWritable(
            new DenseDoubleVector(vals)));
      }
      writer.close();
      br.close();
    } catch (IOException e1) {
      e1.printStackTrace();
    }

    try {
      int iteration = 1000;
      double learningRate = 0.4;
      double momemtumWeight = 0.2;
      double regularizationWeight = 0.01;

      // train the model
      SmallLayeredNeuralNetwork ann = new SmallLayeredNeuralNetwork();
      ann.setLearningRate(learningRate);
      ann.setMomemtumWeight(momemtumWeight);
      ann.setRegularizationWeight(regularizationWeight);
      ann.addLayer(featureDimension, false,
          FunctionFactory.createDoubleFunction("Sigmoid"));
      ann.addLayer(featureDimension, false,
          FunctionFactory.createDoubleFunction("Sigmoid"));
      ann.addLayer(labelDimension, true,
          FunctionFactory.createDoubleFunction("Sigmoid"));
      ann.setCostFunction(FunctionFactory
          .createDoubleDoubleFunction("CrossEntropy"));
      ann.setModelPath(MODEL_PATH);

      Map<String, String> trainingParameters = new HashMap<String, String>();
      trainingParameters.put("tasks", "2");
      trainingParameters.put("training.max.iterations", "" + iteration);
      trainingParameters.put("training.batch.size", "300");
      trainingParameters.put("convergence.check.interval", "1000");
      ann.train(conf, sequenceTrainingDataPath, trainingParameters);

    } catch (Exception e) {
      e.printStackTrace();
    }
  }
}
