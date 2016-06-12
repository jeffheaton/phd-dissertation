package com.jeffheaton.dissertation.experiments.payloads;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;

/**
 * Created by jeff on 6/12/16.
 */
public interface ExperimentPayload {
    void init(ThreadedRunner theRunner, ExperimentTask theTask);
    void run();
}
