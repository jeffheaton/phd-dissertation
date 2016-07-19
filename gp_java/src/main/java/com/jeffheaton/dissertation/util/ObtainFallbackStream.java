package com.jeffheaton.dissertation.util;

import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import org.encog.EncogError;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

/**
 * Created by Jeff on 6/7/2016.
 */
public class ObtainFallbackStream implements ObtainInputStream {
    private String datasetName;

    public ObtainFallbackStream(String theDatasetName) {
        this.datasetName = theDatasetName;
    }


    @Override
    public InputStream obtain() {
        File target = null;
        final InputStream istream = this.getClass().getResourceAsStream("/"+this.datasetName);
        if (istream == null) {
            try {
                target = new File(DissertationConfig.getInstance().getDataPath(),this.datasetName);
                return new FileInputStream(target);
            } catch (FileNotFoundException e) {
                throw new EncogError("Cannot access data set, make sure the resources are available: " + target);
            }

        }
        return istream;
    }
}
