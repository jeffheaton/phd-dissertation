package com.jeffheaton.dissertation.experiments.ex2;

import com.jeffheaton.dissertation.experiments.ExperimentResult;
import com.jeffheaton.dissertation.experiments.data.AnalyzeEngineeredDataset;
import com.jeffheaton.dissertation.experiments.data.SyntheticDatasets;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.experiments.manager.FileBasedTaskManager;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import com.jeffheaton.dissertation.experiments.manager.ThreadedRunner;
import com.jeffheaton.dissertation.experiments.report.GenerateComparisonReport;
import com.jeffheaton.dissertation.util.NewSimpleEarlyStoppingStrategy;
import com.jeffheaton.dissertation.util.Transform;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationReLU;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.mathutil.randomize.XaiverRandomizer;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.error.CrossEntropyErrorFunction;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.sgd.StochasticGradientDescent;
import org.encog.util.Format;
import org.encog.util.Stopwatch;

import java.io.File;

/**
 * Created by jeff on 5/10/16.
 */
public class PerformExperiment2 {

    public static void addDataSet(TaskQueueManager manager, boolean regression, String filename, String target) {
        String type = regression ? "r":"c";
        manager.addTaskCycles("exp2",filename,"neural-"+type+":"+target,null,5);
        if( regression ) {
            manager.addTaskCycles("exp2", filename, "gp-r:" + target, null, 5);
        }
    }


    public static void main(String[] args) {
        Stopwatch sw = new Stopwatch();
        sw.start();

        ErrorCalculation.setMode(ErrorCalculationMode.RMS);
        TaskQueueManager manager = new FileBasedTaskManager();

        manager.removeAll();
        addDataSet(manager,true,"auto-mpg.csv","mpg");
        addDataSet(manager,false,"iris.csv","species");
        addDataSet(manager,true,"abalone.csv","sex");
        addDataSet(manager,true,"bupa.csv","sector");

        ThreadedRunner runner = new ThreadedRunner(manager);
        runner.setVerbose(false);
        runner.startup();
        manager.blockUntilDone(60);
        runner.shutdown();

        GenerateComparisonReport report = new GenerateComparisonReport(manager);
        File reportFile = new File(DissertationConfig.getInstance().getProjectPath(),"report-exp2.csv");
        report.report(reportFile, 60);

        Encog.getInstance().shutdown();
        sw.stop();
        System.out.println("Total runtime: " + Format.formatTimeSpan((int)(sw.getElapsedMilliseconds()/1000)));
        ErrorCalculation.setMode(ErrorCalculationMode.LOGLOSS);
    }
}
