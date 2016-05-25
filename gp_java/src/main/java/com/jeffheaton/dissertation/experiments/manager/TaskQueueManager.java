package com.jeffheaton.dissertation.experiments.manager;

/**
 * Created by jeff on 5/16/16.
 */
public interface TaskQueueManager {
    String addTask(String name, String dataset, String algorithm, int cycle);
    void remoteTask(String key);
    void removeAll();
    ExperimentTask requestTask();

    void addTaskCycles(String exp1, String s, String neural, int i);
}
