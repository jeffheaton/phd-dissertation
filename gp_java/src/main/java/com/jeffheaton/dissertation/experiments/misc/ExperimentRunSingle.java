package com.jeffheaton.dissertation.experiments.misc;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;

/**
 * Created by Jeff on 6/8/2016.
 */
public class ExperimentRunSingle {

    public static void main(String[] args) {
        /*ExperimentTask task = new ExperimentTask(
                "test",
                "feature_eng.csv",
                "gp-r:diff-y0",
                "diff-x0,diff-x1",
                0);
        task.run();*/

        ExperimentTask task = new ExperimentTask(
                "test",
                "feature_eng.csv",
                "gp-r:ratio_diff-y0",
                "ratio_diff-x0,ratio_diff-x1,ratio_diff-x2,ratio_diff-x3",
                0);
        task.run();


    }

}
