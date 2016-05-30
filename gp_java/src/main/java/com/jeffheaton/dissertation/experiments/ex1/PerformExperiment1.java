package com.jeffheaton.dissertation.experiments.ex1;

import com.jeffheaton.dissertation.experiments.ExperimentResult;
import com.jeffheaton.dissertation.experiments.data.SyntheticDatasets;
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
public class PerformExperiment1 {

    public static void main(String[] args) {
        Stopwatch sw = new Stopwatch();
        sw.start();

        ErrorCalculation.setMode(ErrorCalculationMode.MSE);
        //MLDataSet dataset = SyntheticDatasets.generateDiffRatio();
        MLDataSet dataset = SyntheticDatasets.generatePolynomial();

        //runExperiment(dataset);

        TaskQueueManager manager = new FileBasedTaskManager(new File("/Users/jeff/temp/runjob"),"mac");

        /*manager.removeAll();
        manager.addTaskCycles("exp1","autompg","neural",5);
        manager.addTaskCycles("exp1","autompg","gp",5);*/

        GenerateComparisonReport report = new GenerateComparisonReport(manager);
        report.report(new File("/Users/jeff/temp/runjob/report.csv"), 60);

/*       ThreadedRunner runner = new ThreadedRunner(manager);
        runner.startup();
        manager.blockUntilDone(60);
        runner.shutdown();*/

        Encog.getInstance().shutdown();
        sw.stop();
        System.out.println("Total runtime: " + Format.formatTimeSpan((int)(sw.getElapsedMilliseconds()/1000)));

    }
}
