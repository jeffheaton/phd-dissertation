package com.jeffheaton.dissertation;

import com.jeffheaton.dissertation.experiments.ex1.PerformExperiment1;
import com.jeffheaton.dissertation.experiments.ex2.PerformExperiment2;
import com.jeffheaton.dissertation.experiments.ex3.PerformExperiment3;
import com.jeffheaton.dissertation.experiments.misc.ExperimentAutoFeature;
import com.jeffheaton.dissertation.experiments.misc.ExperimentGPFile;
import com.jeffheaton.dissertation.experiments.misc.ExperimentNeuralXOR;
import com.jeffheaton.dissertation.experiments.misc.ExperimentSimpleGP;

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
        } else if( args[0].equalsIgnoreCase("experiment-1")) {
            (new PerformExperiment1()).main(null);
        } else if( args[0].equalsIgnoreCase("experiment-2")) {
            (new PerformExperiment2()).main(null);
        } else if( args[0].equalsIgnoreCase("experiment-3")) {
            (new PerformExperiment3()).main(null);
        }
    }
}
