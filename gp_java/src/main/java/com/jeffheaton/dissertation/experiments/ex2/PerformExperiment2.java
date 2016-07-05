package com.jeffheaton.dissertation.experiments.ex2;

import com.jeffheaton.dissertation.experiments.AbstractExperiment;
import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.FileBasedTaskManager;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;

import java.io.File;
import java.util.List;

/**
 * Experiment 2: For the dissertation algorithm to be effective, engineered features from this algorithm must enhance
 * neural network accuracy.  To measure this performance, a number of public and synthetic data sets will evaluate it.
 * It is necessary to collect a baseline RMSE or log loss of a deep neural network with that data set that
 * receives no help from the dissertation algorithm.  The neural network topology, or hyper-parameters, will
 * be determined experimentally.  It is important that the topology include enough hidden neurons that the data
 * set can be learned with reasonable accuracy.
 */
public class PerformExperiment2 extends AbstractExperiment {

    public static void addDataSet(TaskQueueManager manager, DatasetInfo info) {
        String type = info.isRegression() ? "r":"c";
        manager.addTaskCycles("exp2",info.getName(),"neural-"+type+":"+info.getTarget(),null,5);
        if( info.isRegression() || info.getTargetElements()<3 ) {
            manager.addTaskCycles("exp2", info.getName(), "gp-r:" + info.getTarget(), null, 5);
        }
    }


    public static void main(String[] args) {
        PerformExperiment2 experiment = new PerformExperiment2();
        experiment.run();
    }

    @Override
    public String getName() {
        return "experiment-2";
    }

    @Override
    protected void internalRun() {
        File path = createPath();

        TaskQueueManager manager = new FileBasedTaskManager(path);

        manager.removeAll();

        List<DatasetInfo> datasets = ExperimentDatasets.getInstance().getDatasetsForExperiment(getName());
        for(DatasetInfo info: datasets) {
            addDataSet(manager,info);
        }

        ThreadedRunner runner = new ThreadedRunner(manager);
        runner.setVerbose(false);
        runner.startup();
        manager.blockUntilDone(60);
        runner.shutdown();

        GenerateAggregateReport report = new GenerateAggregateReport(manager);
        File reportFile = new File(path,"report-exp2.csv");
        report.report(reportFile, 60);
    }
}
