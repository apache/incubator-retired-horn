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

import org.apache.hama.commons.math.DoubleDoubleFunction;

/**
 * The cross entropy cost function.
 * 
 * <pre>
 * cost(t, y) = - t * log(y) - (1 - t) * log(1 - y),
 * where t denotes the target value, y denotes the estimated value.
 * </pre>
 */
public class CrossEntropy extends DoubleDoubleFunction {

  private static final double epsilon = 1e-8;
  
  @Override
  public double apply(double target, double actual) {
    double adjustedTarget = (target == 0 ? 0.000001 : target);
    adjustedTarget = (target == 1.0 ? 0.999999 : adjustedTarget);
    double adjustedActual = (actual == 0 ? 0.000001 : actual);
    adjustedActual = (actual == 1 ? 0.999999 : adjustedActual);
    
    return -target * Math.log(Math.max(actual, epsilon)) - (1 - target)
        * Math.log(Math.max(1 - actual, epsilon));
    // return -adjustedTarget * Math.log(adjustedActual) - (1 - adjustedTarget) *  Math.log(adjustedActual);
  }
  
  @Override
  public double applyDerivative(double target, double actual) {
    double adjustedTarget = (target == 0 ? 0.000001 : target);
    adjustedTarget = (target == 1.0 ? 0.999999 : adjustedTarget);
    double adjustedActual = (actual == 0 ? 0.000001 : actual);
    adjustedActual = (actual == 1 ? 0.999999 : adjustedActual);
    
    return -adjustedTarget / adjustedActual + (1 - adjustedTarget)
        / (1 - adjustedActual);
  }

}
