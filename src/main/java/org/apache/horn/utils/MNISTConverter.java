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

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hama.HamaConfiguration;
import org.apache.hama.commons.io.FloatVectorWritable;
import org.apache.hama.commons.math.DenseFloatVector;

public class MNISTConverter {

  private static int PIXELS = 28 * 28;

  private static float rescale(float x) {
    return 1 - (255 - x) / 255;
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 3) {
      System.out.println("Usage: <TRAINING_DATA> <LABELS_DATA> <OUTPUT_PATH>");
      System.out
          .println("ex) train-images.idx3-ubyte train-labels.idx1-ubyte /tmp/mnist.seq");
      System.exit(1);
    }

    String training_data = args[0];
    String labels_data = args[1];
    String output = args[2];

    HamaConfiguration conf = new HamaConfiguration();
    conf.set("dfs.block.size", "11554432");
    FileSystem fs = FileSystem.get(conf);

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

    System.out.println("Writing " + count + " samples on " + output);

    byte[][] images = new byte[count][PIXELS];
    byte[] labels = new byte[count];
    for (int n = 0; n < count; n++) {
      imagesIn.readFully(images[n]);
      labels[n] = labelsIn.readByte();
    }

    @SuppressWarnings("deprecation")
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path(
        output), LongWritable.class, FloatVectorWritable.class);

    for (int i = 0; i < count; i++) {
      float[] vals = new float[PIXELS + 10];
      for (int j = 0; j < PIXELS; j++) {
        vals[j] = rescale((images[i][j] & 0xff));
      }
      int label = (labels[i] & 0xff);
      // embedding to one-hot vector
      for (int j = 0; j < 10; j++) {
        if (j == label)
          vals[PIXELS + j] = 1.0f;
        else
          vals[PIXELS + j] = 0.0f;
      }

      writer.append(new LongWritable(), new FloatVectorWritable(
          new DenseFloatVector(vals)));
    }

    imagesIn.close();
    labelsIn.close();
    writer.close();
  }

}
