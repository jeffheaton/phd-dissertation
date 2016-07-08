package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;

public abstract class AbstractExperimentPayload implements ExperimentPayload {

    private boolean verbose;

    public boolean isVerbose() {
        return this.verbose;
    }

    public void setVerbose(boolean theVerbose) {
        this.verbose = theVerbose;
    }
}
p