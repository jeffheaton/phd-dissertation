package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import org.encog.Encog;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSoftMax;

import java.util.Arrays;

public class ExperimentRunSingle {

    public static void main(String[] args) {
        /*ExperimentTask task = new ExperimentTask(
                "test",
                "feature_eng.csv",
                "gp-r:diff-y0",
                "diff-x0,diff-x1",
                0);
        task.clearLog();
        task.run();*/

        /*ExperimentTask task = new ExperimentTask(
                "test",
                "feature_eng.csv",
                "gp-r:ratio_diff-y0",
                "ratio_diff-x0,ratio_diff-x1,ratio_diff-x2,ratio_diff-x3",
                0);
        task.clearLog();
        task.run();*/

        /*ExperimentTask task = new ExperimentTask(
                "test",
                "iris.csv",
                "neural-c:species",
                null,
                0);
        task.clearLog();
        task.run();*/

        /*ExperimentTask task = new ExperimentTask(
                "test",
                "auto-mpg.csv",
                "neural-r:mpg",
                null,
                0);
        task.clearLog();
        task.run();*/

        ExperimentTask task = new ExperimentTask(
                "test",
                "feature_eng.csv",
                "gp-r:coef_ratio-y0|rmse",
                "coef_ratio-x0,coef_ratio-x1",
                0);
        task.clearLog();
        task.run();




        Encog.getInstance().shutdown();

    }

}
