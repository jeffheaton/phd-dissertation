package com.jeffheaton.dissertation.util;

/**
 * Created by Jeff on 3/31/2016.
 */
public class FeatureRanking implements Comparable<FeatureRanking> {
    private final String name;
    private double totalWeight;
    private double importancePercent;

    public FeatureRanking(String theName) {
        this.name = theName;
    }

    public String getName() {
        return name;
    }

    public void addWeight(double theWeight) {
        this.totalWeight+=theWeight;
    }

    public double getTotalWeight() {
        return totalWeight;
    }

    public void setTotalWeight(double totalWeight) {
        this.totalWeight = totalWeight;
    }

    public double getImportancePercent() {
        return importancePercent;
    }

    public void setImportancePercent(double importancePercent) {
        this.importancePercent = importancePercent;
    }

    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append(getName());
        result.append(", importance:");
        result.append(getImportancePercent());
        result.append(", total weight:");
        result.append(getTotalWeight());
        return result.toString();
    }

    private String GetName() {
        return this.name;
    }

    @Override
    public int compareTo(FeatureRanking o) {
        return Double.compare(getImportancePercent(),o.getImportancePercent());
    }
}
