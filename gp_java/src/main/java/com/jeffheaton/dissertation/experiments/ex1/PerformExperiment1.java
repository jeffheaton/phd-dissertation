package com.jeffheaton.dissertation.experiments.ex1;

import com.jeffheaton.dissertation.experiments.AbstractExperiment;
import com.jeffheaton.dissertation.experiments.data.AnalyzeEngineeredDataset;
import com.jeffheaton.dissertation.experiments.data.DatasetInfo;
import com.jeffheaton.dissertation.experiments.data.ExperimentDatasets;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.FileBasedTaskManager;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;
import org.encog.Encog;

import java.io.File;
import java.util.List;

/**
 * Created by jeff on 5/10/16.
 */
public class PerformExperiment1 extends AbstractExperiment {

    public String getName() {
        return "experiment-1";
    }

    protected void internalRun() {

        File path = DissertationConfig.getInstance().createPath(getName());
        TaskQueueManager manager = new FileBasedTaskManager(path);

        manager.removeAll();
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

            manager.addTaskCycles(getName(),"feature_eng.csv","neural-r:"+info.getTarget(),pred.toString(),5);
            manager.addTaskCycles(getName(),"feature_eng.csv","gp-r:"+info.getTarget(),pred.toString(),5);
        }

        ThreadedRunner runner = new ThreadedRunner(manager);
        runner.setVerbose(false);
        runner.startup();
        manager.blockUntilDone(600);
        runner.shutdown();

        GenerateAggregateReport report = new GenerateAggregateReport(manager);
        File reportFile = new File(DissertationConfig.getInstance().getProjectPath(),"report-exp1.csv");
        report.report(reportFile, 600);
    }


    public static void main(String[] args) {
        PerformExperiment1 experiment = new PerformExperiment1();
        experiment.run();
        Encog.getInstance().shutdown();
    }
}
