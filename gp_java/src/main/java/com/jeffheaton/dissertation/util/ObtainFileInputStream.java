package com.jeffheaton.dissertation.util;

import org.encog.EncogError;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

/**
 * Created by Jeff on 6/7/2016.
 */
public class ObtainFileInputStream implements ObtainInputStream {

    private File file;

    public ObtainFileInputStream(File theFile) {
        this.file = theFile;
    }


    @Override
    public InputStream obtain() {
        final InputStream istream;
        try {
            return new FileInputStream(this.file);
        } catch (FileNotFoundException ex) {
            throw new EncogError(ex);
        }
    }
}
