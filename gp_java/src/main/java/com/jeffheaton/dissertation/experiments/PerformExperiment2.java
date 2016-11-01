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
 * Experiment 2: For the dissertation algorithm to be effective, engineered autofeatures from this algorithm must enhance
 * neural network accuracy.  To measure this performance, a number of public and synthetic data sets will evaluate it.
 * It is necessary to collect a baseline RMSE or log loss of a deep neural network with that data set that
 * receives no help from the dissertation algorithm.  The neural network topology, or hyper-parameters, will
 * be determined experimentally.  It is important that the topology include enough hidden neurons that the data
 * set can be learned with reasonable accuracy.
 */
public class PerformExperiment2 implements AbstractExperiment {

    public void addDataSet(TaskQueueManager manager, DatasetInfo info) {
        String type = info.isRegression() ? "r":"c";
        manager.addTaskCycles(getName(),info.getName(),"neural-"+type+":"+info.getTarget()+"|nrmse",null,
                JeffDissertation.NEURAL_REPEAT_COUNT);
        if( info.isRegression() || info.getTargetElements()<3 ) {
            manager.addTaskCycles(getName(), info.getName(), "gp-r:" + info.getTarget()+"|nrmse", null,
                    JeffDissertation.GENETIC_REPEAT_COUNT);
        }
    }

    @Override
    public String getName() {
        return "experiment-2";
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
        File reportFile = new File(DissertationConfig.getInstance().getProjectPath(),"report-exp2.csv");
        report.report(reportFile, getName(), 600);
    }

    public static void main(String[] args) {
        ExperimentRunner ex = new ExperimentRunner();
        ex.addExperiment(new PerformExperiment2());
        ex.runTasks();
        ex.runReports();
        Encog.getInstance().shutdown();
    }

}
