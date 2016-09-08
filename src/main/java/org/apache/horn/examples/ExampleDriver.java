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

import org.apache.hadoop.util.ProgramDriver;
import org.apache.horn.utils.MNISTConverter;
import org.apache.horn.utils.MNISTEvaluator;

public class ExampleDriver {
  
  public static void main(String args[]) {
    int exitCode = -1;
    ProgramDriver pgd = new ProgramDriver();
    try {
      pgd.addClass(
          "MNISTConverter",
          MNISTConverter.class,
          "A utility program that converts MNIST training and label datasets "
          + "into HDFS sequence file.");
      pgd.addClass("MNISTEvaluator", MNISTEvaluator.class,
          "A utility program that evaluates trained model for the MNIST dataset");
      pgd.addClass(
          "MultiLayerPerceptron",
          MultiLayerPerceptron.class,
          "An example program that trains a multilayer perceptron model from HDFS sequence file.");
      exitCode = pgd.run(args);
    } catch (Throwable e) {
      e.printStackTrace();
    }
    System.exit(exitCode);
  }
}
