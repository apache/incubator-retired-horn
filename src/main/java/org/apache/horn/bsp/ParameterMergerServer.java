package org.apache.horn.bsp;

import com.google.common.base.Preconditions;

import org.apache.hama.commons.math.DoubleMatrix;
import org.mortbay.log.Log;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

public class ParameterMergerServer implements ParameterMerger {
	/* The parameter merge base. */
	protected SmallLayeredNeuralNetwork inMemoryModel;

	/* To terminate or not to terminate. */
	protected AtomicBoolean isConverge;

	/* The number of slave works that request commits. */
	protected int SlaveCount;

	/* After mergeLimit, terminate whether the result is converging or not. */
	protected int mergeLimit;

	/* last n training errors. converging is decided based on the average value of these errors. */
	protected double[] trainingErrors;

	/* If the average of last n training errors is smaller than this value, it is converging. */
	protected double prevAvgTrainingError = Double.MAX_VALUE;

	/* current index for trainingErrors. */
	protected int curTrainingError = 0;

	/* how many merges have been conducted? */
	protected int mergeCount = 0;

	public ParameterMergerServer(SmallLayeredNeuralNetwork inMemoryModel, AtomicBoolean isConverge,
	                             int slaveCount, int mergeLimit, int convergenceCheckInterval) {
		this.inMemoryModel = inMemoryModel;
		this.isConverge = isConverge;
		this.SlaveCount = slaveCount;
		this.mergeLimit = mergeLimit;
		this.trainingErrors = new double[convergenceCheckInterval];
	}

	@Override
	public long getProtocolVersion(String s, long l) throws IOException {
		return ParameterMerger.versionID;
	}

	@Override
	public SmallLayeredNeuralNetworkMessage merge(double trainingError, DoubleMatrix[] weightUpdates,
	                                              DoubleMatrix[] prevWeightUpdates) {
		Preconditions.checkArgument(weightUpdates.length == prevWeightUpdates.length);

		Log.info(String.format("Start merging: %d.\n", this.mergeCount));

		if (!this.isConverge.get()) {
			for (int i = 0; i < weightUpdates.length; ++i) {
				weightUpdates[i] = weightUpdates[i].divide(this.SlaveCount);
				prevWeightUpdates[i] = prevWeightUpdates[i].divide(this.SlaveCount);
			}

			synchronized (inMemoryModel) {
				this.inMemoryModel.updateWeightMatrices(weightUpdates);
				this.inMemoryModel.setPrevWeightMatrices(prevWeightUpdates);

				// add trainingError to trainingErrors
				this.trainingErrors[this.curTrainingError++] = trainingError;

				// check convergence
				if (this.trainingErrors.length == this.curTrainingError) {
					double curAvgTrainingError = 0.0;
					for (int i = 0; i < this.curTrainingError; ++i) {
						curAvgTrainingError += this.trainingErrors[i];
					}
					curAvgTrainingError /= this.trainingErrors.length;

					if (prevAvgTrainingError < curAvgTrainingError) {
						this.isConverge.set(true);
					} else {
						// update
						prevAvgTrainingError = curAvgTrainingError;
						this.curTrainingError = 0;
					}
				}

				if (++this.mergeCount == this.mergeLimit) {
					this.isConverge.set(true);
				}
			}
		}

		return new SmallLayeredNeuralNetworkMessage(
				0, this.isConverge.get(), this.inMemoryModel.getWeightMatrices(),
				this.inMemoryModel.getPrevMatricesUpdates());
	}
}
