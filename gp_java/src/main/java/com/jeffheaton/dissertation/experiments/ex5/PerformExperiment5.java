package com.jeffheaton.dissertation.experiments.ex5;

import com.jeffheaton.dissertation.experiments.AbstractExperiment;
import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.FileBasedTaskManager;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;
import com.jeffheaton.dissertation.experiments.report.GenerateSimpleReport;

import java.io.File;
import java.util.List;

/**
 * Created by jeff on 6/25/16.
 */
public class PerformExperiment5  extends AbstractExperiment {
    public static void addDataSet(TaskQueueManager manager, DatasetInfo info) {
        String type = info.isRegression() ? "r":"c";
        manager.addTask("exp5",info.getName(),"importance-"+type+":"+info.getTarget(),null,1);
    }


    public static void main(String[] args) {
        PerformExperiment5 ex = new PerformExperiment5();
        ex.run();
    }

    @Override
    public String getName() {
        return "experiment-5";
    }

    @Override
    protected void internalRun() {
        File path = createPath();
        TaskQueueManager manager = new FileBasedTaskManager(path);

        manager.removeAll();
        List<DatasetInfo> datasets = ExperimentDatasets.getInstance().getDatasetsForExperiment(getName());
        for(DatasetInfo info: datasets) {
            if( info.isRegression() || info.getTargetElements()<3 ) {
                addDataSet(manager, info);
            }
        }

        ThreadedRunner runner = new ThreadedRunner(manager);
        runner.setVerbose(false);
        runner.startup();
        manager.blockUntilDone(60);
        runner.shutdown();

        GenerateSimpleReport report = new GenerateSimpleReport(manager);
        File reportFile = new File(path,"report-exp5.csv");
        report.report(reportFile, 60);
    }
}
