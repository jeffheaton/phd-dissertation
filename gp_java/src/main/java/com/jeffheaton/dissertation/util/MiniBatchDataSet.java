package com.jeffheaton.dissertation.util;

import java.util.Iterator;

import org.encog.EncogError;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;

public class MiniBatchDataSet implements MLDataSet {

    private MLDataSet dataset;
    private int currentIndex;
    private int batchSize;
    private GenerateRandom random;

    public MiniBatchDataSet(MLDataSet theDataset, GenerateRandom theRandom) {
        this.dataset = theDataset;
        this.random = theRandom;
        setBatchSize(500);
    }

    /**
     * @param theSize Set the batch size, but not larger than the dataset.
     */
    public void setBatchSize(int theSize) {
        this.batchSize = Math.min(theSize,this.dataset.size());
    }

    public int getBatchSize() {
        return this.batchSize;
    }

    @Override
    public Iterator<MLDataPair> iterator() {
        throw new EncogError("Unsupported.");
    }

    @Override
    public int getIdealSize() {
        return this.dataset.getIdealSize();
    }

    @Override
    public int getInputSize() {
        return this.dataset.getInputSize();
    }

    @Override
    public boolean isSupervised() {
        return this.dataset.isSupervised();
    }

    @Override
    public long getRecordCount() {
        return this.batchSize;
    }

    @Override
    public void getRecord(long index, MLDataPair pair) {
        this.dataset.getRecord((index+this.currentIndex)%this.dataset.size(), pair);
    }

    @Override
    public MLDataSet openAdditional() {
        return this;
    }

    @Override
    public void add(MLData data1) {
        throw new EncogError("Unsupported.");
    }

    @Override
    public void add(MLData inputData, MLData idealData) {
        throw new EncogError("Unsupported.");

    }

    @Override
    public void add(MLDataPair inputData) {
        throw new EncogError("Unsupported.");

    }

    @Override
    public void close() {

    }

    @Override
    public int size() {
        return this.batchSize;
    }

    @Override
    public MLDataPair get(int index) {
        return this.dataset.get((index+this.currentIndex)%this.dataset.size());
    }

    public void advance() {
        this.currentIndex = (this.currentIndex+this.batchSize)%this.dataset.size();
    }

    public int getCurrentIndex() {
        return currentIndex;
    }

    public void setCurrentIndex(int currentIndex) {
        this.currentIndex = currentIndex;
    }
}
