package com.jeffheaton.dissertation.experiments;

import com.jeffheaton.dissertation.features.AutoEngineerFeatures;
import com.jeffheaton.dissertation.util.Transform;
import org.encog.mathutil.error.ErrorCalculation;
import org.encog.mathutil.error.ErrorCalculationMode;
import org.encog.ml.data.MLDataSet;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.InputStream;

/**
 * Created by Jeff on 4/2/2016.
 */
public class ExperimentAutoFeature {

    public static void main(String[] args) {
        ErrorCalculation.setMode(ErrorCalculationMode.RMS);

        InputStream is = ExperimentNeuralAutoMPG.loadDatasetMPG();
        ReadCSV csv = new ReadCSV(is, true, CSVFormat.EG_FORMAT.DECIMAL_POINT);
        MLDataSet trainingSet = ExperimentNeuralAutoMPG.loadCSV(csv, new int[]{1, 2, 3, 4, 5, 6, 7}, new int[]{0});
        Transform.interpolate(trainingSet);
        Transform.zscore(trainingSet);

        AutoEngineerFeatures auto = new AutoEngineerFeatures(trainingSet);
        auto.run();
    }
}
