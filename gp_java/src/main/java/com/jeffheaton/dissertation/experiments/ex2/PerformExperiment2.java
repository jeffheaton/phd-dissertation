package com.jeffheaton.dissertation.experiments.ex2;

import com.jeffheaton.dissertation.experiments.AbstractExperiment;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.FileBasedTaskManager;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.experiments.report.GenerateComparisonReport;

import java.io.File;

/**
 * Created by jeff on 5/10/16.
 */
public class PerformExperiment2 extends AbstractExperiment {

    public static void addDataSet(TaskQueueManager manager, boolean regression, String filename, String target) {
        String type = regression ? "r":"c";
        manager.addTaskCycles("exp2",filename,"neural-"+type+":"+target,null,5);
        if( regression ) {
            manager.addTaskCycles("exp2", filename, "gp-r:" + target, null, 5);
        }
    }


    public static void main(String[] args) {
        PerformExperiment2 experiment = new PerformExperiment2();
        experiment.run();
    }

    @Override
    public String getName() {
        return "experiment-2";
    }

    @Override
    protected void internalRun() {
        File path = createPath();

        TaskQueueManager manager = new FileBasedTaskManager(path);

        manager.removeAll();
        addDataSet(manager,true,"auto-mpg.csv","mpg");
        addDataSet(manager,false,"iris.csv","species");
        addDataSet(manager,false,"abalone.csv","sex");
        addDataSet(manager,true,"bupa.csv","selector");
        addDataSet(manager,true,"covtype.csv","cover_type");
        addDataSet(manager,true,"ecoli.csv","sequence");
        addDataSet(manager,false,"forestfires.csv","area");
        addDataSet(manager,true,"glass.csv","type");
        addDataSet(manager,false,"hepatitis.csv","class");
        addDataSet(manager,false,"horse-colic.csv","outcome");
        addDataSet(manager,false,"housing.csv","crim");
        addDataSet(manager,false,"pima-indians-diabetes.csv","class");
        addDataSet(manager,false,"wcbreast_wdbc.csv","diagnosis");
        addDataSet(manager,false,"wcbreast_wpbc.csv","outcome");
        addDataSet(manager,false,"wine.csv","class");
        addDataSet(manager,false,"crx.csv","a16");


        ThreadedRunner runner = new ThreadedRunner(manager);
        runner.setVerbose(false);
        runner.startup();
        manager.blockUntilDone(60);
        runner.shutdown();

        GenerateComparisonReport report = new GenerateComparisonReport(manager);
        File reportFile = new File(path,"report-exp2.csv");
        report.report(reportFile, 60);
    }
}
