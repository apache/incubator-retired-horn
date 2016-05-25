package org.apache.horn.core;

import org.apache.hama.commons.math.FloatVector;

public class FloatFeatureTransformer {

  public FloatFeatureTransformer() {
  }

  /**
   * Directly return the original features.
   */
  public FloatVector transform(FloatVector originalFeatures) {
    return originalFeatures;
  }

}
