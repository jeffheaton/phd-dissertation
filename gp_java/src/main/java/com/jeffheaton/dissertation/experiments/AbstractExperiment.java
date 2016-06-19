package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.util.Format;
import org.encog.util.Stopwatch;

import java.io.File;

/**
 * Created by jeff on 6/19/16.
 */
public abstract class AbstractExperiment {

    public abstract String getName();
    protected abstract void internalRun();

    protected File createPath() {
        File experimentsRoot = new File(DissertationConfig.getInstance().getProjectPath(), "experiment-results");
        if( !experimentsRoot.exists() ) {
            experimentsRoot.mkdir();
        }
        File experimentRoot = new File(experimentsRoot,getName());
        if( !experimentRoot.exists() ) {
            experimentRoot.mkdir();
        }

        // Clear this experiment
        File[] files = experimentRoot.listFiles();
        if( files!=null ) {
            for (File file : experimentRoot.listFiles()) {
                if (file.isFile()) {
                    file.delete();
                }
            }
        }

        return experimentRoot;
    }

    public void run() {
        System.out.println("Beginning " + getName());
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        Stopwatch sw = new Stopwatch();
        sw.start();

        internalRun();

        System.out.println(getName() + ", total runtime: " + Format.formatTimeSpan((int)(sw.getElapsedMilliseconds()/1000)));
        sw.stop();
    }
}
