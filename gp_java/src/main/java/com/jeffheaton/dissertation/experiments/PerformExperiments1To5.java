package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.experiments.manager.ExperimentRunner;
import org.encog.Encog;

public class PerformExperiments1To5 {
    public static void main(String[] args) {
        ExperimentRunner ex = new ExperimentRunner();
        ex.addExperiment(new PerformExperiment1());
        ex.addExperiment(new PerformExperiment2());
        ex.addExperiment(new PerformExperiment3());
        ex.addExperiment(new PerformExperiment4());
        ex.addExperiment(new PerformExperiment5());
        ex.runTasks();
        ex.runReports();
        Encog.getInstance().shutdown();
    }
}
