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

public class ExclusiveOrConverter {

  private static int STEPS = 2;
  private static int LABELS = 2;

  public static void main(String[] args) throws Exception {

    HamaConfiguration conf = new HamaConfiguration();
    conf.set("dfs.block.size", "11554432");
    FileSystem fs = FileSystem.get(conf);
    String output = "semi.seq";
    
    @SuppressWarnings("deprecation")
    SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, new Path(
        output), LongWritable.class, FloatVectorWritable.class);

    float[][] vals = {{0,0,1,0},{0,1,0,1},{1,0,0,1},{1,1,1,0}};

    for (int i = 0; i < vals.length; i++) {
      writer.append(new LongWritable(), new FloatVectorWritable(
          new DenseFloatVector(vals[i])));
    }
    writer.close();
  }

}
