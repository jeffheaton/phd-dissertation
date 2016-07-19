package com.jeffheaton.dissertation.experiments.manager;

import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.util.Format;
import org.encog.util.Stopwatch;

import java.io.File;

/**
 * Created by jeff on 6/19/16.
 */
public abstract interface AbstractExperiment {
    String getName();
    void registerTasks(TaskQueueManager manager);
    void runReport(TaskQueueManager manager);
}
