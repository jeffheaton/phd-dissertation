package com.jeffheaton.dissertation.data;

import com.jeffheaton.dissertation.util.Loader;
import com.jeffheaton.dissertation.util.Transform;
import org.encog.EncogError;
import org.encog.ml.data.MLDataSet;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.IOException;
import java.io.InputStream;

/**
 * Created by Jeff on 4/3/2016.
 */
public class AutoMPG extends PreloadedDataset {

    private static AutoMPG instance;

    @Override
    public InputStream openStream() {
        final InputStream istream = this.getClass().getResourceAsStream("/auto-mpg.csv");
        if (istream == null) {
            System.out.println("Cannot access data set, make sure the resources are available.");
            System.exit(1);
        }
        return istream;
    }

    @Override
    public MLDataSet loadData() {
        InputStream is = openStream();
        ReadCSV csv = new ReadCSV(is,true, CSVFormat.EG_FORMAT.DECIMAL_POINT);
        MLDataSet dataset = Loader.loadCSV(csv,new int[] {1,2,3,4,5,6,7}, new int[] {0});
        Transform.interpolate(dataset);
        try {
            is.close();
        } catch (IOException e) {
            throw new EncogError(e);
        }
        return dataset;
    }

    public static AutoMPG getInstance() {
        if( instance==null ) {
            instance = new AutoMPG();
        }
        return instance;
    }
}
