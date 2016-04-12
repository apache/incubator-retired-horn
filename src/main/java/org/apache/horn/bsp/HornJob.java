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
package org.apache.horn.bsp;

import java.io.IOException;

import org.apache.hama.HamaConfiguration;
import org.apache.hama.bsp.BSPJob;
import org.apache.hama.commons.math.Function;
import org.apache.horn.trainer.Neuron;
import org.apache.horn.trainer.Trainer;

public class HornJob extends BSPJob {

  public HornJob(HamaConfiguration conf, Class<?> exampleClass)
      throws IOException {
    super(conf);
    this.setBspClass(Trainer.class);
    this.setJarByClass(exampleClass);
  }

  public void setDouble(String name, double value) {
    conf.setDouble(name, value);
  }

  @SuppressWarnings("rawtypes")
  public void addLayer(int i, Class<? extends Neuron> class1,
      Class<? extends Function> class2) {
    // TODO Auto-generated method stub

  }

  public void setCostFunction(Class<? extends Function> class1) {
    // TODO Auto-generated method stub

  }

  public void setMaxIteration(int n) {
    this.conf.setInt("horn.max.iteration", n);
  }

}
