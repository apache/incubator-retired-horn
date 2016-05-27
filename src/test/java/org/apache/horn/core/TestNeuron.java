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

import org.apache.hadoop.io.FloatWritable;
import org.apache.horn.funcs.CrossEntropy;
import org.apache.horn.funcs.Sigmoid;

public class TestNeuron extends TestCase {
  private static float learningrate = 0.1f;
  private static float bias = -1;
  private static float theta = 0.8f;

  public static class MyNeuron extends
      Neuron<Synapse<FloatWritable, FloatWritable>> {

    @Override
    public void forward(
        Iterable<Synapse<FloatWritable, FloatWritable>> messages)
        throws IOException {
      float sum = 0;
      for (Synapse<FloatWritable, FloatWritable> m : messages) {
        sum += m.getInput() * m.getWeight();
      }
      sum += (bias * theta);
      System.out.println(new CrossEntropy().apply(0.000001f, 1.0f));
      this.feedforward(new Sigmoid().apply(sum));
    }
    
    @Override
    public void backward(
        Iterable<Synapse<FloatWritable, FloatWritable>> messages)
        throws IOException {
      for (Synapse<FloatWritable, FloatWritable> m : messages) {
        // Calculates error gradient for each neuron
        float gradient = new Sigmoid().applyDerivative(this.getOutput())
            * (m.getDelta() * m.getWeight());

        // Propagates to lower layer
        backpropagate(gradient);

        // Weight corrections
        float weight = learningrate * this.getOutput() * m.getDelta();
        assertEquals(-0.006688235f, weight);
        // this.push(weight);
      }
    }

  }

  public void testProp() throws IOException {
    List<Synapse<FloatWritable, FloatWritable>> x = new ArrayList<Synapse<FloatWritable, FloatWritable>>();
    x.add(new Synapse<FloatWritable, FloatWritable>(0, new FloatWritable(1.0f),
        new FloatWritable(0.5f)));
    x.add(new Synapse<FloatWritable, FloatWritable>(0, new FloatWritable(1.0f),
        new FloatWritable(0.4f)));

    MyNeuron n = new MyNeuron();
    n.forward(x);
    assertEquals(0.5249792f, n.getOutput());

    x.clear();
    x.add(new Synapse<FloatWritable, FloatWritable>(0, new FloatWritable(
        -0.1274f), new FloatWritable(-1.2f)));
    n.backward(x);
  }

}
