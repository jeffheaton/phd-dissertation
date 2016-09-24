package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;

public interface ExperimentPayload {
    boolean isVerbose();
    void setVerbose(boolean theVerbose);
    PayloadReport run(ExperimentTask task);
}
