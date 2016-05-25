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

import org.apache.hama.commons.math.DenseFloatVector;
import org.apache.hama.commons.math.DoubleVector;
import org.apache.hama.commons.math.FloatFunction;
import org.apache.hama.commons.math.FloatVector;
import org.apache.horn.core.IntermediateOutput;

public class SoftMax extends FloatFunction {

  @Override
  public float apply(float value) {
    // it will be handled by intermediate output handler
    return value;
  }

  @Override
  public float applyDerivative(float value) {
    return value * (1f - value);
  }
  
  public static class SoftMaxOutputComputer extends IntermediateOutput {

    @Override
    public FloatVector interlayer(FloatVector output) throws IOException {
      FloatVector expVec = new DenseFloatVector(output.getDimension());
      float sum = 0.0f;
      for(int i = 0; i < output.getDimension(); ++i) {
        float exp = (float) Math.exp(output.get(i));
        sum += exp;
        expVec.set(i, exp);
      }
      // divide by the sum of exponential of the whole vector
      FloatVector softmaxed = expVec.divide(sum);
      return softmaxed;
    }

  }

}
