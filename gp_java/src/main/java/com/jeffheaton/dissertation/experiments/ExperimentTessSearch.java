package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.data.MLDataSet;
import org.encog.util.csv.CSVFormat;

import java.io.File;

/**
 * Created by jheaton on 4/12/2016.
 */
public class ExperimentTessSearch {

    public static void main(String[] args) {
        ExperimentTessSearch file = new ExperimentTessSearch();
        file.process("");
    }

    private void process(String filename) {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        QuickEncodeDataset quick = new QuickEncodeDataset();
        //quick.analyze(new File(filename), 0, true, CSVFormat.EG_FORMAT);
        //quick.dumpFieldInfo();
    }
}
