package com.jeffheaton.dissertation.experiments.manager;

import java.util.List;

public class QueueUtil {
    public static int countError(List<ExperimentTask> tasks) {
        int result = 0;
        for(ExperimentTask task: tasks) {
            if( task.isError()) {
                result++;
            }
        }
        return result;
    }

    public static int countComplete(List<ExperimentTask> tasks) {
        int result = 0;
        for(ExperimentTask task: tasks) {
            if( task.isComplete()) {
                result++;
            }
        }
        return result;
    }
}
