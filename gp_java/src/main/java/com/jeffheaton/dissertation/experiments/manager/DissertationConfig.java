package com.jeffheaton.dissertation.experiments.manager;

import org.encog.EncogError;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Created by Jeff on 6/6/2016.
 */
public final class DissertationConfig {
    public static final String PROPERTIES_FILE = ".dissertation_jheaton";
    private static DissertationConfig instance;
    private File homePath;
    private File propertiesPath;
    private final Properties prop = new Properties();

    private DissertationConfig() {
        this.homePath = new File(System.getProperty("user.home"));
        this.propertiesPath = new File(this.getHomePath(),DissertationConfig.PROPERTIES_FILE);

        InputStream input = null;

        try {
            input = new FileInputStream(this.propertiesPath);

            // load a properties file
            this.prop.load(input);

        } catch (IOException ex) {
            throw new EncogError(ex);
        } finally {
            if (input != null) {
                try {
                    input.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static DissertationConfig getInstance() {
        if( DissertationConfig.instance == null) {
            DissertationConfig.instance = new DissertationConfig();
        }
        return DissertationConfig.instance;
    }

    public File getHomePath() {
        return this.homePath;
    }

    public String getHost() {
        return( prop.getProperty("host"));
    }

    public File getProjectPath() {
        return( new File(prop.getProperty("project")));
    }

    public File getDataPath() {
        return new File(getProjectPath(),"data");
    }

}
