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

import junit.framework.TestCase;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hama.commons.math.DenseFloatVector;
import org.apache.hama.commons.math.FloatVector;
import org.apache.horn.funcs.CrossEntropy;
import org.apache.horn.funcs.Sigmoid;

public class TestNeuron extends TestCase {
  private static float learningrate = 0.1f;
  private static float bias = -1;
  private static float theta = 0.8f;

  public static class MyNeuron extends Neuron {

    @Override
    public void forward(FloatVector input) throws IOException {
      float sum = 0;
      sum += input.multiply(this.getWeightVector()).sum();
      sum += (bias * theta);
      System.out.println(new CrossEntropy().apply(0.000001f, 1.0f));
      this.feedforward(new Sigmoid().apply(sum));
    }

    @Override
    public void backward(
        FloatVector deltaVector)
        throws IOException {
      float delta = this.getWeightVector().multiply(deltaVector).sum();
      /*
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
      */
    }

  }

  public void testProp() throws IOException {
    FloatVector x = new DenseFloatVector(2);
    x.set(0, 1.0f);
    x.set(1, 1.0f);
    FloatVector w = new DenseFloatVector(2);
    w.set(0, 0.5f);
    w.set(1, 0.4f);
    
    MyNeuron n = new MyNeuron();
    n.setWeightVector(w);
    n.forward(x);
    assertEquals(0.5249792f, n.getOutput());

    /*
    x.clear();
    x.add(new Synapse<FloatWritable, FloatWritable>(0, new FloatWritable(-0.1274f), new FloatWritable(-1.2f)));
    n.backward(x);
    */
  }

}
