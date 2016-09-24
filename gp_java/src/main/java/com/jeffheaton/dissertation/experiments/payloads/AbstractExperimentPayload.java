package com.jeffheaton.dissertation.experiments.payloads;

public abstract class AbstractExperimentPayload implements ExperimentPayload {

    private boolean verbose;

    public boolean isVerbose() {
        return this.verbose;
    }

    public void setVerbose(boolean theVerbose) {
        this.verbose = theVerbose;
    }
}