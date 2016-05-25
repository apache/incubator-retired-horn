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
 * Square error cost function.
 * 
 * <pre>
 * cost(t, y) = 0.5 * (t - y) &circ; 2
 * </pre>
 */
public class SquaredError extends FloatFloatFunction {

  @Override
  /**
   * {@inheritDoc}
   */
  public float apply(float target, float actual) {
    float diff = target - actual;
    return (0.5f * diff * diff);
  }

  @Override
  /**
   * {@inheritDoc}
   */
  public float applyDerivative(float target, float actual) {
    return actual - target;
  }

}
