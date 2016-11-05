package com.jeffheaton.dissertation.experiments.manager;

import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.util.Format;
import org.encog.util.Stopwatch;

import java.util.ArrayList;
import java.util.List;

public class ExperimentRunner {
    private Stopwatch sw;
    private List<AbstractExperiment> experimentList = new ArrayList<>();
    private List<DatasetInfo> datasets = new ArrayList<>();
    private TaskQueueManager manager;

    public void addExperiment(AbstractExperiment experiment) {
        this.experimentList.add(experiment);
    }

    public void runTasks(boolean actuallyRun) {
        this.sw = new Stopwatch();
        sw.start();

        System.out.println("Analyzing datasets...");
        ExperimentDatasets.getInstance();
        System.out.println("Analysis complete.");
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        this.manager = new FileBasedTaskManager(DissertationConfig.getInstance().getProjectPath());

        if( actuallyRun ) {
            manager.removeAll();

            for (AbstractExperiment experiment : this.experimentList) {
                experiment.registerTasks(manager);
            }

            ThreadedRunner runner = new ThreadedRunner(manager);
            runner.setVerbose(false);
            runner.startup();
            manager.blockUntilDone(600);
            runner.shutdown();
        }
    }

    public void runReports() {

        System.out.println("Running reports.");

        for(AbstractExperiment experiment: this.experimentList) {
            experiment.runReport(this.manager);
        }

        sw.stop();
        System.out.println("Total runtime: " + Format.formatTimeSpan((int)(this.sw.getElapsedMilliseconds()/1000)));
    }
}
