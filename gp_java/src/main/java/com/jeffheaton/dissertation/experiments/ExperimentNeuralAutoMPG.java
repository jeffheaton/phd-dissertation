package com.jeffheaton.dissertation.experiments;

import org.encog.ml.data.MLDataSet;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.File;
import java.io.InputStream;


public class ExperimentNeuralAutoMPG {

    public static MLDataSet loadCSV(File path, boolean headers, CSVFormat format, int[] input, int[] ideal) {
        ReadCSV csv = new ReadCSV(path,true,format);
        return null;
    }

    public InputStream loadDatasetMPG() {
        final InputStream istream = this.getClass().getResourceAsStream("/auto-mpg.csv");
        if (istream == null) {
            System.out.println("Cannot access data set, make sure the resources are available.");
            System.exit(1);
        }
        return istream;
    }

    public void run() {
        InputStream is = loadDatasetMPG();
        ReadCSV csv = new ReadCSV(is,true,CSVFormat.EG_FORMAT.DECIMAL_POINT);
        while(csv.next()) {
            System.out.println(csv.get(0));
        }
    }


    public static void main(String[] args) {
        ExperimentNeuralAutoMPG prg = new ExperimentNeuralAutoMPG();
        prg.run();
    }
}
