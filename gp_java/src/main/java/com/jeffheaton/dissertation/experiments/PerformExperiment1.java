package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.*;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;
import org.encog.Encog;

import java.io.File;
import java.util.List;

public class PerformExperiment1 implements AbstractExperiment {

    @Override
    public String getName() {
        return "experiment-1";
    }

    @Override
    public void registerTasks(TaskQueueManager manager) {
        List<DatasetInfo> datasets = ExperimentDatasets.getInstance().getDatasetsForExperiment(getName());

        for(DatasetInfo info: datasets) {

            // Build comma separated list
            StringBuilder pred = new StringBuilder();
            for(String str: info.getPredictors()) {
                if( pred.length()>0 ) {
                    pred.append(",");
                }
                pred.append(str);
            }

            manager.addTaskCycles(getName(),"feature_eng.csv","neural-r:"+info.getTarget()+"|rmse",pred.toString(),
                    JeffDissertation.NEURAL_REPEAT_COUNT);
            manager.addTaskCycles(getName(),"feature_eng.csv","gp-r:"+info.getTarget()+"|rmse",pred.toString(),
                    JeffDissertation.GENETIC_REPEAT_COUNT);
        }
    }

    @Override
    public void runReport(TaskQueueManager manager) {
        GenerateAggregateReport report = new GenerateAggregateReport(manager);
        File reportFile = new File(DissertationConfig.getInstance().getProjectPath(),"report-exp1.csv");
        report.report(reportFile, getName(), 600);
    }

    public static void main(String[] args) {
        ExperimentRunner ex = new ExperimentRunner();
        ex.addExperiment(new PerformExperiment1());
        ex.runTasks();
        ex.runReports();
        Encog.getInstance().shutdown();
    }
}
