package com.jeffheaton.dissertation.util;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.util.csv.ReadCSV;

/**
 * Created by Jeff on 4/3/2016.
 */
public class Loader {
    public static MLDataSet loadCSV(ReadCSV csv, int[] input, int[] ideal) {
        MLDataSet result = new BasicMLDataSet();
        while(csv.next()) {
            MLData inputItem = new BasicMLData(input.length);
            MLData idealItem = new BasicMLData(ideal.length);
            MLDataPair pair = new BasicMLDataPair(inputItem,idealItem);

            // Read input
            int idx = 0;
            for(int i:input) {
                inputItem.setData(idx++,csv.getDouble(i));
            }

            // Read ideal
            idx = 0;
            for(int i:ideal) {
                idealItem.setData(idx++,csv.getDouble(i));
            }

            result.add(pair);

        }
        return result;
    }

}
