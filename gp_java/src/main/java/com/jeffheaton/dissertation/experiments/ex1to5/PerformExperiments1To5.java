package com.jeffheaton.dissertation.experiments.ex1to5;

import com.jeffheaton.dissertation.experiments.ex1.PerformExperiment1;
import com.jeffheaton.dissertation.experiments.ex2.PerformExperiment2;
import com.jeffheaton.dissertation.experiments.ex3.PerformExperiment3;
import com.jeffheaton.dissertation.experiments.ex4.PerformExperiment4;
import com.jeffheaton.dissertation.experiments.ex5.PerformExperiment5;
import org.encog.Encog;
import org.encog.util.Format;
import org.encog.util.Stopwatch;

/**
 * Created by jeffh on 7/5/2016.
 */
public class PerformExperiments1To5 {
    public static void main(String[] args) {
        Stopwatch sw = new Stopwatch();
        sw.start();

        (new PerformExperiment1()).run();
        (new PerformExperiment2()).run();
        (new PerformExperiment3()).run();
        (new PerformExperiment4()).run();
        (new PerformExperiment5()).run();

        System.out.println("Total runtime: " + Format.formatTimeSpan((int)(sw.getElapsedMilliseconds()/1000)));
        sw.stop();
        Encog.getInstance().shutdown();
    }
}
