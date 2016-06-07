package com.jeffheaton.dissertation.experiments.manager;

import org.encog.EncogError;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by jeff on 6/7/16.
 */
public class ParseModelType {

    private final String name;
    private final String type;
    private final String target;

    public ParseModelType(String str) {
        try {
            String regexp = "([A-Za-z]+)-+([A-Za-z]+):+([A-Za-z]+)";

            Pattern pattern = Pattern.compile(regexp);
            Matcher matcher = pattern.matcher(str);
            matcher.find();
            this.name = matcher.group(1);
            this.type = matcher.group(2);
            this.target = matcher.group(3);
        } catch(IllegalStateException ex) {
            throw new EncogError("Invalid model type: " + str, ex);
        }
    }

    public String getName() {
        return name;
    }

    public String getType() {
        return type;
    }

    public String getTarget() {
        return target;
    }

    public boolean isNeuralNetwork() {
        return this.name.toLowerCase().equals("neural");
    }

    public boolean isGeneticProgram() {
        return this.name.toLowerCase().equals("gp");
    }

    public boolean isRegression() {
        return this.type.toLowerCase().charAt(0)=='r';
    }
}
