package com.jeffheaton.dissertation.experiments.manager;

import java.util.List;

public interface TaskQueueManager {
    ExperimentTask addTask(String name, String dataset, String model, String predictors, int cycle);
    void removeTask(String key);
    void removeAll();
    ExperimentTask requestTask(int maxWaitSeconds);
    void addTaskCycles(String name, String dataset, String model, String predictors, int cycles);
    void reportDone(ExperimentTask task,int maxWaitSeconds);
    void blockUntilDone(int maxWaitSeconds);
    void reportError(ExperimentTask task, Exception ex, int maxWaitSeconds);
    List<ExperimentTask> getQueue(int maxWaitSeconds);
}
