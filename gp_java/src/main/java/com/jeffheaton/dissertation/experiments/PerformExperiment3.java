package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.*;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;
import org.encog.Encog;

import java.io.File;
import java.util.List;


public class PerformExperiment3 implements AbstractExperiment {

    public void addDataSet(TaskQueueManager manager, DatasetInfo info) {
        String type = info.isRegression() ? "r":"c";
        manager.addTaskCycles(getName(),info.getName(),"ensemble-"+type+":"+info.getTarget(),null,5);
    }

    @Override
    public String getName() {
        return "experiment-3";
    }

    @Override
    public void registerTasks(TaskQueueManager manager) {
        List<DatasetInfo> datasets = ExperimentDatasets.getInstance().getDatasetsForExperiment(getName());
        for(DatasetInfo info: datasets) {
            if( info.isRegression() || info.getTargetElements()<3 ) {
                addDataSet(manager, info);
            }
        }
    }

    @Override
    public void runReport(TaskQueueManager manager) {
        GenerateAggregateReport report = new GenerateAggregateReport(manager);
        File reportFile = new File(DissertationConfig.getInstance().getProjectPath(),"report-exp3.csv");
        report.report(reportFile, getName(), 600);
    }

    public static void main(String[] args) {
        ExperimentRunner ex = new ExperimentRunner();
        ex.addExperiment(new PerformExperiment3());
        ex.runTasks();
        ex.runReports();
        Encog.getInstance().shutdown();
    }
}
