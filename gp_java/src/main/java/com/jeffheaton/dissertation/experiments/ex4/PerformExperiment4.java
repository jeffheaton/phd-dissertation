package com.jeffheaton.dissertation.experiments.ex4;

import com.jeffheaton.dissertation.experiments.AbstractExperiment;
import com.jeffheaton.dissertation.experiments.manager.FileBasedTaskManager;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.experiments.report.GenerateAggregateReport;
import com.jeffheaton.dissertation.experiments.report.GenerateSimpleReport;

import java.io.File;

/**
 * Created by jeff on 6/25/16.
 */
public class PerformExperiment4  extends AbstractExperiment {
    public static void addDataSet(TaskQueueManager manager, boolean regression, String filename, String target) {
        String type = regression ? "r":"c";
        manager.addTask("exp4",filename,"patterns-"+type+":"+target,null,1);
    }


    public static void main(String[] args) {
        PerformExperiment4 ex = new PerformExperiment4();
        ex.run();
    }

    @Override
    public String getName() {
        return "experiment-4";
    }

    @Override
    protected void internalRun() {
        File path = createPath();
        TaskQueueManager manager = new FileBasedTaskManager(path);

        manager.removeAll();
        addDataSet(manager,true,"auto-mpg.csv","mpg");
        ////addDataSet(manager,false,"iris.csv","species");
        ////addDataSet(manager,false,"abalone.csv","sex");
        addDataSet(manager,true,"bupa.csv","selector");
        /////addDataSet(manager,true,"covtype.csv","cover_type"); -- too slow!
        addDataSet(manager,true,"ecoli.csv","sequence");
        ////addDataSet(manager,false,"forestfires.csv","area");
        addDataSet(manager,true,"glass.csv","type");
        ////addDataSet(manager,false,"hepatitis.csv","class");
        ////addDataSet(manager,false,"horse-colic.csv","outcome");
        ////addDataSet(manager,false,"housing.csv","crim");
        ////addDataSet(manager,false,"pima-indians-diabetes.csv","class");
        ////addDataSet(manager,false,"wcbreast_wdbc.csv","diagnosis");
        ////addDataSet(manager,false,"wcbreast_wpbc.csv","outcome");
        ////addDataSet(manager,false,"wine.csv","class");
        ////addDataSet(manager,false,"crx.csv","a16");

        ThreadedRunner runner = new ThreadedRunner(manager);
        runner.setVerbose(false);
        runner.startup();
        manager.blockUntilDone(60);
        runner.shutdown();

        GenerateSimpleReport report = new GenerateSimpleReport(manager);
        File reportFile = new File(path,"report-exp4.csv");
        report.report(reportFile, 60);
    }
}
