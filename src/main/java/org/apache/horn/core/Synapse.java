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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;

/**
 * Message wrapper for a propagating message
 */
public class Synapse<M extends Writable, W extends Writable> implements
    Writable {

  FloatWritable message;
  FloatWritable weight;
  FloatWritable prevWeight;

  public Synapse(FloatWritable message, FloatWritable weight) {
    this.message = message;
    this.weight = weight;
  }

  public Synapse(FloatWritable message, FloatWritable weight, FloatWritable prevWeight) {
    this.message = message;
    this.weight = weight;
    this.prevWeight = prevWeight;
  }
  
  /**
   * @return the activation or error message
   */
  public float getMessage() {
    return message.get();
  }

  public float getInput() {
    // returns the input
    return message.get();
  }
  
  public float getDelta() {
    // returns the delta
    return message.get();
  }
  
  public float getWeight() {
    return weight.get();
  }
  
  public float getPrevWeight() {
    return prevWeight.get();
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    message.readFields(in);
    weight.readFields(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    message.write(out);
    weight.write(out);
  }

}