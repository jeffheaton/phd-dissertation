package com.jeffheaton.dissertation.experiments.manager;

/**
 * Created by jeff on 5/16/16.
 */
public interface TaskQueueManager {
    ExperimentTask addTask(String name, String dataset, String algorithm, int cycle);
    void removeTask(String key);
    void removeAll();
    ExperimentTask requestTask(int maxWaitSeconds);
    void addTaskCycles(String exp1, String s, String neural, int i);
    void reportDone(ExperimentTask task,int maxWaitSeconds);
    void blockUntilDone(int maxWaitSeconds);
}
