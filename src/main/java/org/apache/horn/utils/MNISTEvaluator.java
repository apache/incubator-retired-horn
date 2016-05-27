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
package org.apache.horn.utils;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import org.apache.hama.HamaConfiguration;
import org.apache.hama.commons.math.DenseFloatVector;
import org.apache.hama.commons.math.FloatVector;
import org.apache.horn.core.LayeredNeuralNetwork;

public class MNISTEvaluator {

  private static int PIXELS = 28 * 28;

  private static float rescale(float x) {
    return 1 - (255 - x) / 255;
  }

  public static void main(String[] args) throws IOException {
    if (args.length < 3) {
      System.out.println("Usage: <TRAINED_MODEL> <TEST_IMAGES> <TEST_LABELS>");
      System.out
          .println("ex) /tmp/model t10k-images.idx3-ubyte t10k-labels.idx1-ubyte");
      System.exit(1);
    }

    String modelPath = args[0];
    String training_data = args[1];
    String labels_data = args[2];

    DataInputStream imagesIn = new DataInputStream(new FileInputStream(
        new File(training_data)));
    DataInputStream labelsIn = new DataInputStream(new FileInputStream(
        new File(labels_data)));

    imagesIn.readInt(); // Magic number
    int count = imagesIn.readInt();
    labelsIn.readInt(); // Magic number
    labelsIn.readInt(); // Count
    imagesIn.readInt(); // Rows
    imagesIn.readInt(); // Cols

    HamaConfiguration conf = new HamaConfiguration();
    LayeredNeuralNetwork ann = new LayeredNeuralNetwork(conf, modelPath);

    int correct = 0;
    int total = 0;

    for (int n = 0; n < count; n++) {
      byte[] vector = new byte[PIXELS];
      imagesIn.readFully(vector);
      int label = (labelsIn.readByte() & 0xff);

      FloatVector instance = new DenseFloatVector(PIXELS);
      for (int j = 0; j < PIXELS; j++) {
        instance.set(j, rescale((vector[j] & 0xff)));
      }
      FloatVector result = ann.getOutput(instance);

      if (getNumber(result) == label) {
        correct++;
      }
      total++;
    }

    System.out.println("Accuracy: " + ((double) correct / total));

    imagesIn.close();
    labelsIn.close();
  }

  private static int getNumber(FloatVector result) {
    double max = 0;
    int index = -1;
    for (int x = 0; x < result.getLength(); x++) {
      double curr = result.get(x);
      if (max < curr) {
        max = curr;
        index = x;
      }
    }
    return index;
  }
}
