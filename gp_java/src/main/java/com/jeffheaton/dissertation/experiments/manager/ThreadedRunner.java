package com.jeffheaton.dissertation.experiments.manager;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jeff on 5/16/16.
 */
public class ThreadedRunner {
    private final List<ExperimentTask> tasks = new ArrayList<>();

    public void addTask(ExperimentTask theTask) {
        this.tasks.add(theTask);
    }

    public List<ExperimentTask> getTasks() {
        return tasks;
    }
}
