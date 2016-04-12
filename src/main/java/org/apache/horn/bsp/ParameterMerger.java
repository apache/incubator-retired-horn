package org.apache.horn.bsp;

import org.apache.hama.ipc.VersionedProtocol;

public interface ParameterMerger extends VersionedProtocol {
  long versionID = 1L;

  SmallLayeredNeuralNetworkMessage merge(SmallLayeredNeuralNetworkMessage msg);

}
