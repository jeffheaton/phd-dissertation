package com.jeffheaton.dissertation.experiments.manager;

import org.encog.EncogError;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ParseModelType {

    private final String name;
    private final String type;
    private final String target;
    private final String error;

    public ParseModelType(String str) {
        try {
            String regexp = "^([A-Za-z]+)-+([A-Za-z]+):+([0-9A-Za-z-_]+)\\|([A-Za-z]+)";

            Pattern pattern = Pattern.compile(regexp);
            Matcher matcher = pattern.matcher(str);
            matcher.find();
            this.name = matcher.group(1);
            this.type = matcher.group(2);
            this.target = matcher.group(3);
            this.error = matcher.group(4)==null?"nrmse":matcher.group(4);
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

    public boolean isEnsemble() { return this.name.toLowerCase().equals("ensemble"); }

    public boolean isRegression() {
        return this.type.toLowerCase().charAt(0)=='r';
    }

    public boolean isPatterns() { return this.name.toLowerCase().equals("patterns"); }

    public boolean isImportance() { return this.name.toLowerCase().equals("importance"); }

    public boolean isAutoFeature() { return this.name.toLowerCase().equals("autofeature"); }

    public String getError() { return this.error; }

    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append(getName());
        result.append("-");
        result.append(getType());
        result.append(":");
        result.append("target");
        result.append("|");
        result.append(getError());
        return result.toString();
    }
}
