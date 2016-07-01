package com.jeffheaton.dissertation.experiments.manager;

import com.jeffheaton.dissertation.experiments.payloads.*;
import com.jeffheaton.dissertation.util.*;
import org.encog.EncogError;
import org.encog.ml.data.MLDataSet;
import org.encog.util.csv.CSVFormat;

import java.util.Random;

/**
 * Created by jeff on 5/16/16.
 */
public class ExperimentTask implements Runnable {

    private final String name;
    private final String algorithm;
    private final String datasetFilename;
    private final int cycle;
    private String status = "queued";
    private int iterations;
    private double result;
    private int elapsed;
    private QuickEncodeDataset quick;
    private MLDataSet dataset;
    private String predictors;
    private String info;
    private boolean regression;
    private ThreadedRunner owner;

    public ExperimentTask(String theName, String theDataset, String theAlgorithm, String thePredictors, int theCycle) {
        this.name = theName;
        this.datasetFilename = theDataset;
        this.algorithm = theAlgorithm;
        this.cycle = theCycle;
        this.predictors = thePredictors;
    }

    public String getName() {
        return name;
    }

    public String getAlgorithm() {
        return algorithm;
    }

    public String getDatasetFilename() {
        return this.datasetFilename;
    }

    public int getCycle() {
        return cycle;
    }

    public String getKey() {
        StringBuilder result = new StringBuilder();
        result.append(this.name);
        result.append("|");
        result.append(this.algorithm);
        result.append("|");
        result.append(this.datasetFilename);
        result.append("|");
        result.append(this.cycle);
        return result.toString();
    }

    private void loadDataset(boolean singleFieldCatagorical, String target) {
        ObtainInputStream source = new ObtainFallbackStream(this.datasetFilename);
        this.quick = new QuickEncodeDataset(singleFieldCatagorical,false);
        this.dataset = quick.process(source, target, this.predictors, true, CSVFormat.EG_FORMAT);
    }

    public void run() {
        ParseModelType model = new ParseModelType(this.algorithm);
        this.regression = model.isRegression();

        PayloadReport report = null;

        if (model.isNeuralNetwork()) {
            loadDataset(false,model.getTarget());
            ExperimentPayload payload = new PayloadNeuralFit();
            payload.setVerbose(this.owner==null||this.owner.isVerbose());
            report = payload.run(this.quick.getFieldNames(),this.dataset,this.regression);
        } else if (model.isGeneticProgram()) {
            loadDataset(true,model.getTarget());
            ExperimentPayload payload = new PayloadGeneticFit();
            payload.setVerbose(this.owner==null||this.owner.isVerbose());
            report = payload.run(this.quick.getFieldNames(),this.dataset,this.regression);
        } else if (model.isEnsemble() ) {
            loadDataset(true,model.getTarget());
            ExperimentPayload payload = new PayloadEnsembleGP();
            payload.setVerbose(this.owner==null||this.owner.isVerbose());
            report = payload.run(this.quick.getFieldNames(),this.dataset,this.regression);
        } else if (model.isPatterns() ) {
            loadDataset(true,model.getTarget());
            ExperimentPayload payload = new PayloadPatterns();
            payload.setVerbose(this.owner==null||this.owner.isVerbose());
            report = payload.run(this.quick.getFieldNames(),this.dataset,this.regression);
        } else if (model.isImportance() ) {
            loadDataset(true,model.getTarget());
            ExperimentPayload payload = new PayloadImportance();
            payload.setVerbose(this.owner==null||this.owner.isVerbose());
            report = payload.run(this.quick.getFieldNames(),this.dataset,this.regression);
        } else {
            throw new EncogError("Unknown algorithm: " + this.algorithm);
        }

        if(report!=null) {
            this.elapsed = report.getElapsed();
            this.result = report.getResult();
            this.iterations = report.getIteration();
            setInfo(report.getComment());
        }

        if(this.owner!=null) {
            this.owner.reportComplete(this);
        }

    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("[ExperimentTask:");
        result.append(getKey());
        result.append("]");
        return result.toString();
    }

    public boolean isQueued() {
        return this.status.equalsIgnoreCase("queued");
    }

    public void claim(String owner) {
        this.status = "running-" + owner;
    }

    public void reportDone(String theOwner) {
        this.status = "done-" + theOwner;
    }

    public boolean isComplete() {
        return this.status.startsWith("done") || this.status.startsWith("error");
    }

    public boolean isError() {
        return this.status.startsWith("error");
    }

    public void reportError(String owner, Exception ex) {
        this.status = "error-" + owner;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public double getResult() {
        return result;
    }

    public void setResult(double result) {
        this.result = result;
    }

    public int getElapsed() {
        return elapsed;
    }

    public void setElapsed(int elapsed) {
        this.elapsed = elapsed;
    }

    public ThreadedRunner getOwner() {
        return owner;
    }

    public void setOwner(ThreadedRunner owner) {
        this.owner = owner;
    }

    public String getPredictors() {
        return this.predictors;
    }

    public String getInfo() {
        return info;
    }

    public void setInfo(String info) {
        this.info = info;
    }

    public MLDataSet getDataset() {
        return dataset;
    }

    public boolean isRegression() {
        return this.regression;
    }

    public QuickEncodeDataset getQuick() {
        return this.quick;
    }

}
