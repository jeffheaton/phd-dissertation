package com.jeffheaton.dissertation.data;

import org.encog.ml.data.MLDataSet;

import java.io.InputStream;

/**
 * Created by Jeff on 4/3/2016.
 */
public abstract class PreloadedDataset {
    abstract InputStream openStream();
    abstract MLDataSet loadData();
}
