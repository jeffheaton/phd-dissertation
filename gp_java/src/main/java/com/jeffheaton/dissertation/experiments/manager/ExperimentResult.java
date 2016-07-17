package com.jeffheaton.dissertation.experiments.manager;

import org.encog.util.Format;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by jeff on 5/11/16.
 */
public class ExperimentResult {
    private final String name;
    private final List<Double> results = new ArrayList<>();
    private final List<Long> elapsed = new ArrayList<>();

    public ExperimentResult(String theName) {
        this.name = theName;
    }

    public void addResult(double theResult, long theElapsed) {
        this.results.add(theResult);
        this.elapsed.add(theElapsed);
    }

    public String getName() {
        return name;
    }

    public List<Double> getResults() {
        return this.results;
    }

    public List<Long> getElapsed() {
        return this.elapsed;
    }

    /**
     * Find the best attempt (minimum or maximum).
     * @param minimize Is the best attempt considered to be the minimum.
     * @return The best attempt, or -1 of no attempts were made.
     */
    public int getBestIndex(boolean minimize) {
        int best = -1;
        for(int i=0;i<this.results.size();i++) {
            if( best==-1 ) {
                best = i;
            } else {
                if( minimize ) {
                    if (this.results.get(i) < this.results.get(best)) {
                        best = i;
                    }
                } else {
                    if (this.results.get(i) > this.results.get(i)) {
                        best = i;
                    }
                }
            }
        }
        return best;
    }

    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("[name=");
        result.append(getName());
        result.append(",cycles=");
        result.append(getResults().size());
        result.append(",bestResult=");
        int best = getBestIndex(true);
        result.append(getResults().get(best));
        result.append(",elapsedTime=");
        result.append(Format.formatTimeSpan((getElapsed().get(best)).intValue()/1000));
        result.append("]");
        return result.toString();
    }
}
