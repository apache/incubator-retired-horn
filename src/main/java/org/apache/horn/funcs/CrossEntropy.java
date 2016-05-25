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

import org.apache.hama.commons.math.FloatFloatFunction;

/**
 * The cross entropy cost function.
 * 
 * <pre>
 * cost(t, y) = - t * log(y) - (1 - t) * log(1 - y),
 * where t denotes the target value, y denotes the estimated value.
 * </pre>
 */
public class CrossEntropy extends FloatFloatFunction {

  private static final float epsilon = 1e-8f;

  @Override
  public float apply(float target, float actual) {
    return -target * (float) Math.log(Math.max(actual, epsilon)) - (1 - target)
        * (float) Math.log(Math.max(1 - actual, epsilon));
  }

  @Override
  public float applyDerivative(float target, float actual) {
    float adjustedTarget = (target == 0 ? 0.000001f : target);
    adjustedTarget = (target == 1.0 ? 0.999999f : adjustedTarget);
    float adjustedActual = (actual == 0 ? 0.000001f : actual);
    adjustedActual = (actual == 1 ? 0.999999f : adjustedActual);

    return -adjustedTarget / adjustedActual + (1 - adjustedTarget)
        / (1 - adjustedActual);
  }

}
