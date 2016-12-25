package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.*;
import com.jeffheaton.dissertation.experiments.report.GenerateSimpleReport;
import org.encog.Encog;

import java.io.File;
import java.util.List;

public class PerformExperiment5  implements AbstractExperiment {
    public void addDataSet(TaskQueueManager manager, DatasetInfo info) {
        String type = info.isRegression() ? "r":"c";
        manager.addTask(getName(),info.getName(),"importance-"+type+":"+info.getTarget()+"|rmse",null,1);
    }

    @Override
    public String getName() {
        return "experiment-5";
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
        report.setIFieldNames("IMP-W","IMP-P");
        File reportFile = new File(DissertationConfig.getInstance().getProjectPath(),"report-exp5.csv");
        report.report(reportFile, getName(), 600);
    }

    public static void main(String[] args) {
        ExperimentRunner ex = new ExperimentRunner();
        ex.addExperiment(new PerformExperiment5());
        ex.runTasks(true);
        ex.runReports();
        Encog.getInstance().shutdown();
    }

}
