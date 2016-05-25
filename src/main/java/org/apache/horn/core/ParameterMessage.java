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

import org.apache.hadoop.io.Writable;
import org.apache.hama.commons.io.FloatMatrixWritable;
import org.apache.hama.commons.math.DenseFloatMatrix;
import org.apache.hama.commons.math.FloatMatrix;

/**
 * ParameterMessage transmits the messages between workers and parameter
 * servers during the training of neural networks.
 * 
 */
public class ParameterMessage implements Writable {

  protected float trainingError;
  protected FloatMatrix[] curMatrices;
  protected FloatMatrix[] prevMatrices;
  protected boolean converge;

  public ParameterMessage() {
    this.converge = false;
    this.trainingError = 0.0f;
  }

  public ParameterMessage(float trainingError, boolean converge,
      FloatMatrix[] weightMatrices, FloatMatrix[] prevMatrices) {
    this.trainingError = trainingError;
    this.converge = converge;
    this.curMatrices = weightMatrices;
    this.prevMatrices = prevMatrices;
  }

  @Override
  public void readFields(DataInput input) throws IOException {
    trainingError = input.readFloat();
    converge = input.readBoolean();
    boolean hasCurMatrices = input.readBoolean();
    if(hasCurMatrices) {
      int numMatrices = input.readInt();
      curMatrices = new DenseFloatMatrix[numMatrices];
      // read matrice updates
      for (int i = 0; i < curMatrices.length; ++i) {
        curMatrices[i] = (DenseFloatMatrix) FloatMatrixWritable.read(input);
      }
    }
    
    boolean hasPrevMatrices = input.readBoolean();
    if (hasPrevMatrices) {
      int numMatrices = input.readInt();
      prevMatrices = new DenseFloatMatrix[numMatrices];
      // read previous matrices updates
      for (int i = 0; i < prevMatrices.length; ++i) {
        prevMatrices[i] = (DenseFloatMatrix) FloatMatrixWritable.read(input);
      }
    }
  }

  @Override
  public void write(DataOutput output) throws IOException {
    output.writeFloat(trainingError);
    output.writeBoolean(converge);
    if (curMatrices == null) {
      output.writeBoolean(false);
    } else {
      output.writeBoolean(true);
      output.writeInt(curMatrices.length);
      for (FloatMatrix matrix : curMatrices) {
        FloatMatrixWritable.write(matrix, output);
      }
    }
    
    if (prevMatrices == null) {
      output.writeBoolean(false);
    } else {
      output.writeBoolean(true);
      output.writeInt(prevMatrices.length);
      for (FloatMatrix matrix : prevMatrices) {
        FloatMatrixWritable.write(matrix, output);
      }
    }
  }

  public double getTrainingError() {
    return trainingError;
  }

  public void setTrainingError(float trainingError) {
    this.trainingError = trainingError;
  }

  public boolean isConverge() {
    return converge;
  }

  public void setConverge(boolean converge) {
    this.converge = converge;
  }

  public FloatMatrix[] getCurMatrices() {
    return curMatrices;
  }

  public void setMatrices(FloatMatrix[] curMatrices) {
    this.curMatrices = curMatrices;
  }

  public FloatMatrix[] getPrevMatrices() {
    return prevMatrices;
  }

  public void setPrevMatrices(FloatMatrix[] prevMatrices) {
    this.prevMatrices = prevMatrices;
  }

}
