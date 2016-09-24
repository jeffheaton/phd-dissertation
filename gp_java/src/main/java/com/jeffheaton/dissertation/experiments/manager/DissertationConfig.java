package com.jeffheaton.dissertation.experiments.manager;

import org.encog.EncogError;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

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

    public int getThreads() {
        if( prop.containsKey("threads")) {
            return Integer.parseInt(prop.getProperty("threads").trim());
        } else {
            return Runtime.getRuntime().availableProcessors();
        }
    }

    public File getPath(String name) {
        File experimentsRoot = new File(getProjectPath(), "experiment-results");
        if( !experimentsRoot.exists() ) {
            experimentsRoot.mkdir();
        }
        File experimentRoot = new File(experimentsRoot,name);
        if( !experimentRoot.exists() ) {
            experimentRoot.mkdir();
        }

        return experimentRoot;
    }

    public File createPath(String name) {
        File experimentRoot = getPath(name);

        // Clear this experiment
        File[] files = experimentRoot.listFiles();
        if( files!=null ) {
            for (File file : experimentRoot.listFiles()) {
                if (file.isFile()) {
                    file.delete();
                }
            }
        }
        return experimentRoot;
    }

    public boolean isVerboseForced() {
        return prop.containsKey("verbose");
    }

    public boolean getVerbose() {
        if( prop.containsKey("verbose")) {
            String str = prop.getProperty("threads").trim();
            return "Yy1".indexOf(str.trim().charAt(0))!=-1;
        } else {
            return false;
        }
    }

}
