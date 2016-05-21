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
package org.apache.horn.funcs;

import java.io.IOException;

import org.apache.hama.commons.math.DenseDoubleVector;
import org.apache.hama.commons.math.DoubleFunction;
import org.apache.hama.commons.math.DoubleVector;
import org.apache.horn.core.IntermediateOutput;

public class SoftMax extends DoubleFunction {

  @Override
  public double apply(double value) {
    // it will be handled by intermediate output handler
    return value;
  }

  @Override
  public double applyDerivative(double value) {
    return value * (1d - value);
  }
  
  public static class SoftMaxOutputComputer extends IntermediateOutput {

    @Override
    public DoubleVector interlayer(DoubleVector output) throws IOException {
      DoubleVector expVec = new DenseDoubleVector(output.getDimension());
      double sum = 0.0;
      for(int i = 0; i < output.getDimension(); ++i) {
        double exp = Math.exp(output.get(i));
        sum += exp;
        expVec.set(i, exp);
      }
      // divide by the sum of exponential of the whole vector
      DoubleVector softmaxed = expVec.divide(sum);
      return softmaxed;
    }

  }

}
