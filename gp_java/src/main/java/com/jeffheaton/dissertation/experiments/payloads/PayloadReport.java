package com.jeffheaton.dissertation.experiments.payloads;

public class PayloadReport {
    private final int elapsed;
    private final double result;
    private final int iteration;
    private final String comment;
    private final double resultRaw;
    private final int i1;
    private final int i2;

    public PayloadReport(int elapsed, double result, double resultRaw, int i1, int i2, int iteration, String comment) {
        this.elapsed = elapsed;
        this.result = result;
        this.iteration = iteration;
        this.comment = comment;
        this.resultRaw = resultRaw;
        this.i1 = i1;
        this.i2 = i2;
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

    public double getResultRaw() {
        return resultRaw;
    }

    public int getI1() {
        return i1;
    }

    public int getI2() {
        return i2;
    }
}
