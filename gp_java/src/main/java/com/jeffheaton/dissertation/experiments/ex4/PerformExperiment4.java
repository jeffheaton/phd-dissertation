package com.jeffheaton.dissertation.experiments.ex4;

import com.jeffheaton.dissertation.experiments.AbstractExperiment;
import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.ex2.PerformExperiment2;
import com.jeffheaton.dissertation.experiments.manager.*;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;
import com.jeffheaton.dissertation.experiments.report.GenerateSimpleReport;
import org.encog.Encog;

import java.io.File;
import java.util.List;

/**
 * Created by jeff on 6/25/16.
 */
public class PerformExperiment4 implements AbstractExperiment {
    public void addDataSet(TaskQueueManager manager, DatasetInfo info) {
        String type = info.isRegression() ? "r":"c";
        manager.addTask(getName(),info.getName(),"patterns-"+type+":"+info.getTarget(),null,1);
    }

    @Override
    public String getName() {
        return "experiment-4";
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
        GenerateSimpleReport report = new GenerateSimpleReport(manager);
        File reportFile = new File(DissertationConfig.getInstance().getProjectPath(),"report-exp4.csv");
        report.report(reportFile, 60);
    }

    public static void main(String[] args) {
        ExperimentRunner ex = new ExperimentRunner();
        ex.addExperiment(new PerformExperiment4());
        ex.runTasks();
        ex.runReports();
        Encog.getInstance().shutdown();
    }
}
