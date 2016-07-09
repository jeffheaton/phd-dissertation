package com.jeffheaton.dissertation.experiments.ex1to5;

import com.jeffheaton.dissertation.experiments.ex1.PerformExperiment1;
import com.jeffheaton.dissertation.experiments.ex2.PerformExperiment2;
import com.jeffheaton.dissertation.experiments.ex3.PerformExperiment3;
import com.jeffheaton.dissertation.experiments.ex4.PerformExperiment4;
import com.jeffheaton.dissertation.experiments.ex5.PerformExperiment5;
import com.jeffheaton.dissertation.experiments.manager.ExperimentRunner;
import org.encog.Encog;
import org.encog.util.Format;
import org.encog.util.Stopwatch;

/**
 * Created by jeffh on 7/5/2016.
 */
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
