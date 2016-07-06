package com.jeffheaton.dissertation.experiments.ex3;

import com.jeffheaton.dissertation.experiments.AbstractExperiment;
import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.FileBasedTaskManager;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;

import java.io.File;
import java.util.List;


public class PerformExperiment3 extends AbstractExperiment {

    public void addDataSet(TaskQueueManager manager, DatasetInfo info) {
        String type = info.isRegression() ? "r":"c";
        manager.addTaskCycles(getName(),info.getName(),"ensemble-"+type+":"+info.getTarget(),null,5);
    }


    public static void main(String[] args) {
        PerformExperiment3 ex = new PerformExperiment3();
        ex.run();
    }

    @Override
    public String getName() {
        return "experiment-3";
    }

    @Override
    protected void internalRun() {
        File path = DissertationConfig.getInstance().createPath(getName());
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

        GenerateAggregateReport report = new GenerateAggregateReport(manager);
        File reportFile = new File(path,"report-exp3.csv");
        report.report(reportFile, 60);
    }
}
