package com.jeffheaton.dissertation;

import com.jeffheaton.dissertation.experiments.ExperimentAutoFeature;
import com.jeffheaton.dissertation.experiments.ExperimentGPFile;
import com.jeffheaton.dissertation.experiments.ExperimentNeuralXOR;
import com.jeffheaton.dissertation.experiments.ExperimentSimpleGP;

public class JeffDissertation {
    public static void main(String[] args) {
        if( args[0].equalsIgnoreCase("neural-xor") ) {
            (new ExperimentNeuralXOR()).main(null);
        } else if( args[0].equalsIgnoreCase("simple-gp")) {
            (new ExperimentSimpleGP()).main(null);
        } else if( args[0].equalsIgnoreCase("feature-search")) {
            (new ExperimentAutoFeature()).main(null);
        } else if( args[0].equalsIgnoreCase("file-gp")) {
            (new ExperimentGPFile()).main(null);
        }
    }
}
