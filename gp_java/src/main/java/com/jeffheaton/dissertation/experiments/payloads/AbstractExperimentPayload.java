package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;

/**
 * Created by jeff on 6/12/16.
 */
public abstract class AbstractExperimentPayload implements ExperimentPayload {

    private ThreadedRunner runner;
    private ExperimentTask task;
    private QuickEncodeDataset quick;

    @Override
    public void init(ThreadedRunner theRunner,ExperimentTask theTask) {
        this.runner = theRunner;
        this.task = theTask;
    }

    public ThreadedRunner getRunner() {
        return runner;
    }

    public ExperimentTask getTask() {
        return task;
    }
}
