package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import org.encog.ml.data.MLDataSet;

/**
 * Created by jeff on 6/12/16.
 */
public interface ExperimentPayload {
    boolean isVerbose();
    void setVerbose(boolean theVerbose);
    PayloadReport run(ExperimentTask task);
}
