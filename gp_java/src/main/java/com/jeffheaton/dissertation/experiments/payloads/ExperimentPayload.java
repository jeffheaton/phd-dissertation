package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import org.encog.ml.data.MLDataSet;

public interface ExperimentPayload {
    boolean isVerbose();
    void setVerbose(boolean theVerbose);
    PayloadReport run(ExperimentTask task);
    MLDataSet obtainCommonProcessing(ExperimentTask task);
}
