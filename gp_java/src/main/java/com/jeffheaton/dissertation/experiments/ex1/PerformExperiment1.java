package com.jeffheaton.dissertation.experiments.ex1;

import com.jeffheaton.dissertation.experiments.AbstractExperiment;
import com.jeffheaton.dissertation.experiments.data.AnalyzeEngineeredDataset;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.FileBasedTaskManager;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;
import org.encog.Encog;

import java.io.File;

/**
 * Created by jeff on 5/10/16.
 */
public class PerformExperiment1 extends AbstractExperiment {

    public String getName() {
        return "experiment-1";
    }

    protected void internalRun() {

        TaskQueueManager manager = new FileBasedTaskManager(createPath());

        AnalyzeEngineeredDataset info = new AnalyzeEngineeredDataset();

        manager.removeAll();
        for(String f: info.getFeatures()) {
            manager.addTaskCycles("exp1","feature_eng.csv","neural-r:"+f+"-y0",info.getPredictors(f),5);
            manager.addTaskCycles("exp1","feature_eng.csv","gp-r:"+f+"-y0",info.getPredictors(f),5);
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
