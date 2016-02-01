package org.apache.horn.bsp;

import org.apache.hama.commons.math.DoubleMatrix;
import org.apache.hama.ipc.VersionedProtocol;

public interface ParameterMerger extends VersionedProtocol {
	long versionID = 1L;

	SmallLayeredNeuralNetworkMessage merge(double trainingError,  DoubleMatrix[] weightUpdates, DoubleMatrix[] prevWeightUpdates);
}
