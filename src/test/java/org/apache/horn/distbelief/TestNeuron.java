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
package org.apache.horn.distbelief;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import junit.framework.TestCase;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hama.commons.math.Sigmoid;

public class TestNeuron extends TestCase {
  private static double learningRate = 0.1;

  public static class MyNeuron extends
      Neuron<PropMessage<DoubleWritable, DoubleWritable>> {

    @Override
    public void upward(
        Iterable<PropMessage<DoubleWritable, DoubleWritable>> messages)
        throws IOException {
      double sum = 0;
      for (PropMessage<DoubleWritable, DoubleWritable> m : messages) {
        sum += m.getMessage().get() * m.getWeight().get();
      }
      sum += (-1 * 0.8);

      double output = new Sigmoid().apply(sum);
      this.setOutput(output);
    }

    @Override
    public void downward(
        Iterable<PropMessage<DoubleWritable, DoubleWritable>> messages)
        throws IOException {

      for (PropMessage<DoubleWritable, DoubleWritable> m : messages) {
        // Calculates error gradient for each neuron
        double gradient = this.getOutput() * (1 - this.getOutput())
            * m.getMessage().get() * m.getWeight().get();

        // Propagates to lower layer
        System.out.println(gradient);

        // Weight corrections
        double weight = learningRate * this.getOutput() * m.getMessage().get();
        this.push(weight);
      }
    }

  }

  public void testProp() throws IOException {
    List<PropMessage<DoubleWritable, DoubleWritable>> x = new ArrayList<PropMessage<DoubleWritable, DoubleWritable>>();
    x.add(new PropMessage<DoubleWritable, DoubleWritable>(new DoubleWritable(
        1.0), new DoubleWritable(0.5)));
    x.add(new PropMessage<DoubleWritable, DoubleWritable>(new DoubleWritable(
        1.0), new DoubleWritable(0.4)));

    MyNeuron n = new MyNeuron();
    n.upward(x);
    assertEquals(0.5249791874789399, n.getOutput());

    x.clear();
    x.add(new PropMessage<DoubleWritable, DoubleWritable>(new DoubleWritable(
        -0.1274), new DoubleWritable(-1.2)));
    n.downward(x);
    assertEquals(-0.006688234848481696, n.getUpdate());
  }
}
