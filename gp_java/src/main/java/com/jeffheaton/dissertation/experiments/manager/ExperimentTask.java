package com.jeffheaton.dissertation.experiments.manager;

import com.jeffheaton.dissertation.experiments.payloads.*;
import com.jeffheaton.dissertation.util.*;
import org.encog.EncogError;
import org.encog.util.csv.CSVFormat;

import java.io.*;

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
    private double normalizedResult;
    private int i1;
    private int i2;
    private int elapsed;
    private String predictors;
    private String info;
    private ThreadedRunner owner;
    private ParseModelType modelType;
    private File logFilename;

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
        result.append("_");
        result.append(this.algorithm);
        result.append("_");
        result.append(this.datasetFilename);
        result.append("_");
        result.append(this.cycle);

        return result.toString().replaceAll("\\W+", "-");
    }

    public void run() {
        this.modelType = new ParseModelType(this.algorithm);
        PayloadReport report;
        ExperimentPayload payload = null;

        if (this.modelType.isNeuralNetwork()) {
            payload = new PayloadNeuralFit();
        } else if (this.modelType.isGeneticProgram()) {
            payload = new PayloadGeneticFit();
        } else if (this.modelType.isEnsemble() ) {
            payload = new PayloadEnsembleGP();
        } else if (this.modelType.isPatterns() ) {
            payload = new PayloadPatterns();
        } else if (this.modelType.isImportance() ) {
            payload = new PayloadImportance();
        } else {
            throw new EncogError("Unknown algorithm: " + this.algorithm);
        }

        payload.setVerbose(this.owner==null||this.owner.isVerbose());
        report = payload.run(this);

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

    public void log(String str) {
        if( this.logFilename==null ) {
            this.logFilename = new File(DissertationConfig.getInstance().getPath(this.name),getKey()+".log");
        }

        try {
            Writer writer = new BufferedWriter(new OutputStreamWriter(
                    new FileOutputStream(this.logFilename, true), "UTF-8"));
            writer.write(str+"\r\n");
            writer.close();
        } catch(IOException ex) {
            throw new EncogError(ex);
        }
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

    public ParseModelType getModelType() { return this.modelType; }

    public double getNormalizedResult() {
        return normalizedResult;
    }

    public void setNormalizedResult(double normalizedResult) {
        this.normalizedResult = normalizedResult;
    }

    public int getI1() {
        return i1;
    }

    public void setI1(int i1) {
        this.i1 = i1;
    }

    public int getI2() {
        return i2;
    }

    public void setI2(int i2) {
        this.i2 = i2;
    }
}
