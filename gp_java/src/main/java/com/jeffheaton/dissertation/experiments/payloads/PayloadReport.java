package com.jeffheaton.dissertation.experiments.payloads;

/**
 * Created by jeff on 6/16/16.
 */
public class PayloadReport {
    private final int elapsed;
    private final double result;
    private final int iteration;
    private final String comment;

    public PayloadReport(int elapsed, double result, int iteration, String comment) {
        this.elapsed = elapsed;
        this.result = result;
        this.iteration = iteration;
        this.comment = comment;
    }

    public int getElapsed() {
        return elapsed;
    }

    public double getResult() {
        return result;
    }

    public int getIteration() {
        return iteration;
    }

    public String getComment() {
        return comment;
    }
}
