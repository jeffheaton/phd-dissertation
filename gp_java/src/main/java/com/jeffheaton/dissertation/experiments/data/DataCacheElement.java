package com.jeffheaton.dissertation.experiments.data;

import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.ml.data.MLDataSet;

public class DataCacheElement {

    private QuickEncodeDataset quick;
    private MLDataSet data;

    public DataCacheElement(QuickEncodeDataset theQuick) {
        this.quick = theQuick;
    }

    public MLDataSet getData() {
        if( this.data==null ) {
            this.data = this.quick.generateDataset();
        }

        return this.data;
    }

    public QuickEncodeDataset getQuick() {
        return this.quick;
    }
}
