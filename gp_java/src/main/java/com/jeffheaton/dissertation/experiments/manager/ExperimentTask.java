package com.jeffheaton.dissertation.experiments.manager;

/**
 * Created by jeff on 5/16/16.
 */
public class ExperimentTask implements Runnable {

    private final String name;
    private final String algorithm;
    private final String dataset;
    private final int cycle;

    public ExperimentTask(String theName, String theDataset, String theAlgorithm, int theCycle) {
        this.name = theName;
        this.dataset = theDataset;
        this.algorithm = theAlgorithm;
        this.cycle = theCycle;
    }

    public String getName() {
        return name;
    }

    public String getAlgorithm() {
        return algorithm;
    }

    public String getDataset() {
        return dataset;
    }

    public int getCycle() {
        return cycle;
    }

    public void run() {

    }

}
