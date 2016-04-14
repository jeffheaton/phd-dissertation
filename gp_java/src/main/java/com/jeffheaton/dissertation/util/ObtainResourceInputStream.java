package com.jeffheaton.dissertation.util;

import java.io.FileInputStream;
import java.io.InputStream;

/**
 * Created by jeff on 4/12/16.
 */
public class ObtainResourceInputStream implements ObtainInputStream {

    private String resourceName;

    public ObtainResourceInputStream(String theResourceName) {
        this.resourceName = theResourceName;
    }


    @Override
    public InputStream obtain() {
        final InputStream istream = this.getClass().getResourceAsStream(resourceName);
        if (istream == null) {
            System.out.println("Cannot access data set, make sure the resources are available.");
            System.exit(1);
        }
        return istream;
    }
}
