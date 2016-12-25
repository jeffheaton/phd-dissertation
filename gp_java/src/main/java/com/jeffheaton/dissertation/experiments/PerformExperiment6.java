package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.*;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;
import org.encog.Encog;

import java.io.File;
import java.util.List;

/**
 * Experiment 6: Add description.
 */
public class PerformExperiment6 implements AbstractExperiment {

    public void addDataSet(TaskQueueManager manager, DatasetInfo info) {
        String type = info.isRegression() ? "r":"c";
        manager.addTaskCycles(getName(),info.getName(),"autofeature-"+type+":"+info.getTarget()+"|rmse",null,
                JeffDissertation.NEURAL_REPEAT_COUNT);
    }

    @Override
    public String getName() {
        return "experiment-6";
    }

    @Override
    public void registerTasks(TaskQueueManager manager) {
        List<DatasetInfo> datasets = ExperimentDatasets.getInstance().getDatasetsForExperiment(getName());
        for(DatasetInfo info: datasets) {
            addDataSet(manager,info);
        }
    }

    @Override
    public void runReport(TaskQueueManager manager) {
        GenerateAggregateReport report = new GenerateAggregateReport(manager);
        File reportFile = new File(DissertationConfig.getInstance().getProjectPath(),"report-exp6.csv");
        report.report(reportFile, getName(), 600);
    }

    public static void main(String[] args) {
        ExperimentRunner ex = new ExperimentRunner();
        ex.addExperiment(new PerformExperiment6());
        ex.runTasks(true);
        ex.runReports();
        Encog.getInstance().shutdown();
    }
}
