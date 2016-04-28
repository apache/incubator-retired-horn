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
import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.horn.funcs.CrossEntropy;
import org.apache.horn.funcs.Sigmoid;

public class TestNeuron extends TestCase {
  private static double learningrate = 0.1;
  private static double bias = -1;
  private static double theta = 0.8;

  public static class MyNeuron extends
      Neuron<Synapse<DoubleWritable, DoubleWritable>> {

    @Override
    public void forward(
        Iterable<Synapse<DoubleWritable, DoubleWritable>> messages)
        throws IOException {
      double sum = 0;
      for (Synapse<DoubleWritable, DoubleWritable> m : messages) {
        sum += m.getInput() * m.getWeight();
      }
      sum += (bias * theta);
      System.out.println(new CrossEntropy().apply(0.000001, 1.0));
      this.feedforward(new Sigmoid().apply(sum));
    }
    
    @Override
    public void backward(
        Iterable<Synapse<DoubleWritable, DoubleWritable>> messages)
        throws IOException {
      for (Synapse<DoubleWritable, DoubleWritable> m : messages) {
        // Calculates error gradient for each neuron
        double gradient = new Sigmoid().applyDerivative(this.getOutput())
            * (m.getDelta() * m.getWeight());

        // Propagates to lower layer
        backpropagate(gradient);

        // Weight corrections
        double weight = learningrate * this.getOutput() * m.getDelta();
        assertEquals(-0.006688234848481696, weight);
        // this.push(weight);
      }
    }

  }

  public void testProp() throws IOException {
    List<Synapse<DoubleWritable, DoubleWritable>> x = new ArrayList<Synapse<DoubleWritable, DoubleWritable>>();
    x.add(new Synapse<DoubleWritable, DoubleWritable>(new DoubleWritable(1.0),
        new DoubleWritable(0.5)));
    x.add(new Synapse<DoubleWritable, DoubleWritable>(new DoubleWritable(1.0),
        new DoubleWritable(0.4)));

    MyNeuron n = new MyNeuron();
    n.forward(x);
    assertEquals(0.5249791874789399, n.getOutput());

    x.clear();
    x.add(new Synapse<DoubleWritable, DoubleWritable>(new DoubleWritable(
        -0.1274), new DoubleWritable(-1.2)));
    n.backward(x);
  }

}
