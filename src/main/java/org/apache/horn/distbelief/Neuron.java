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

import org.apache.hadoop.io.Writable;

public abstract class Neuron<M extends Writable> implements NeuronInterface<M> {
  double output;
  double weight;

  public void propagate(double gradient) {
    // TODO Auto-generated method stub
  }

  public void setOutput(double output) {
    this.output = output;
  }

  public double getOutput() {
    return output;
  }

  public void push(double weight) {
    // TODO Auto-generated method stub
    this.weight = weight;
  }

  public double getUpdate() {
    return weight;
  }

}
