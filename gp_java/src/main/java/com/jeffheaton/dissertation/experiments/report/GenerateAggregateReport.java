package com.jeffheaton.dissertation.experiments.report;

import au.com.bytecode.opencsv.CSVWriter;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import org.encog.EncogError;
import org.encog.util.Format;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GenerateAggregateReport {
    private TaskQueueManager manager;
    private Map<String, ReportItem> reportItems = new HashMap<>();

    class ReportCycle {
        private final double result;
        private final int iterations;
        private final int seconds;

        public ReportCycle(double result, int iterations, int seconds) {
            this.result = result;
            this.iterations = iterations;
            this.seconds = seconds;
        }

        public double getResult() {
            return result;
        }

        public int getIterations() {
            return iterations;
        }

        public int getSeconds() {
            return seconds;
        }

        @Override
        public String toString() {
            return "ReportCycle{" +
                    "result=" + result +
                    ", iterations=" + iterations +
                    ", seconds=" + seconds +
                    '}';
        }
    }

    class ReportItem {
        private final String experiment;
        private final String algorithm;
        private final String dataset;
        private List<ReportCycle> cycles = new ArrayList<>();
        private double minResult;
        private double maxResult;
        private double meanResult;
        private double sdevResult;
        private int meanElapsed;

        public ReportItem(String theExperiment, String theAlgorithm, String theDataset) {
            this.experiment = theExperiment;
            this.algorithm = theAlgorithm;
            this.dataset = theDataset;
        }

        public ReportItem(ExperimentTask task) {
            this(task.getName(), task.getAlgorithm(), task.getDatasetFilename());
        }

        public void reportCycle(double result, int iterations, int seconds) {
            this.cycles.add(new ReportCycle(result, iterations, seconds));
        }

        public String getExperiment() {
            return experiment;
        }

        public String getDataset() {
            return dataset;
        }

        public String getAlgorithm() {
            return algorithm;
        }

        public List<ReportCycle> getCycles() {
            return cycles;
        }


        @Override
        public String toString() {
            return "ReportItem{" +
                    "experiment='" + experiment + '\'' +
                    ", dataset='" + dataset + '\'' +
                    ", algorithm='" + algorithm + '\'' +
                    ", cycles=" + cycles +
                    '}';
        }

        public double getMinResult() {
            return minResult;
        }

        public double getMaxResult() {
            return maxResult;
        }

        public void collectStats() {
            boolean first = true;
            double sum = 0;
            int secondSum = 0;
            int count = 0;

            // mean, min, max
            for (ReportCycle cycle : this.cycles) {
                count++;
                if (first) {
                    this.minResult = cycle.getResult();
                    this.maxResult = cycle.getResult();
                    first = false;
                } else {
                    this.minResult = Math.min(this.minResult, cycle.getResult());
                    this.maxResult = Math.max(this.maxResult, cycle.getResult());
                }
                sum += cycle.getResult();
                secondSum += cycle.getSeconds();
            }

            this.meanResult = sum / count;
            this.meanElapsed = secondSum / count;

            // sdev
            sum = 0;
            for (ReportCycle cycle : this.cycles) {
                double d = (cycle.getResult() - this.meanResult);
                sum += d * d;
            }
            this.sdevResult = Math.sqrt(sum);
        }

        public double getMeanResult() {
            return meanResult;
        }

        public double getSDevResult() {
            return sdevResult;
        }

        private int getMeanElapsed() {
            return this.meanElapsed;
        }
    }

    class CycleResult {
        private double result;
    }

    public GenerateAggregateReport(TaskQueueManager theManager) {
        this.manager = theManager;
    }

    private String generateKey(ExperimentTask task) {
        StringBuilder result = new StringBuilder();
        result.append(task.getName());
        result.append(':');
        result.append(task.getAlgorithm());
        result.append(':');
        result.append(task.getDatasetFilename());
        return result.toString();
    }

    private String generateKey(ReportItem item) {
        StringBuilder result = new StringBuilder();
        result.append(item.getExperiment());
        result.append(':');
        result.append(item.getAlgorithm());
        result.append(':');
        result.append(item.getDataset());
        return result.toString();
    }

    private void reportCycle(ExperimentTask task) {
        String key = generateKey(task);
        ReportItem item;
        if (this.reportItems.containsKey(key)) {
            item = this.reportItems.get(key);
        } else {
            item = new ReportItem(task);
            this.reportItems.put(key, item);
        }
        item.reportCycle(task.getResult(), task.getIterations(), task.getElapsed());

    }

    public void report(File file, String name, int max) {
        ReportHolder holder = new ReportHolder();

        List<ExperimentTask> queue = this.manager.getQueue(max);
        for (ExperimentTask task : queue) {
            if( task.getName().equals(name) ) {
                reportCycle(task);
            }
        }


        holder.setHeaders(new String[]{"experiment", "algorithm", "dataset", "min", "max", "mean", "sd", "elapsed" });
        for (String key2 : this.reportItems.keySet()) {
            ReportItem item = this.reportItems.get(key2);
            item.collectStats();

            if( Double.isInfinite(item.getMeanResult()) || Double.isNaN(item.getMeanResult()) ) {
                holder.addLine(new String[]{item.getExperiment(), item.getAlgorithm(), item.getDataset(),
                        "NaN", "NaN", "NaN", "NaN",
                        Format.formatTimeSpan(item.getMeanElapsed())});
            } else {
                holder.addLine(new String[]{item.getExperiment(), item.getAlgorithm(), item.getDataset(),
                        Format.formatDouble(item.getMinResult(), 4), Format.formatDouble(item.getMaxResult(), 4),
                        Format.formatDouble(item.getMeanResult(), 4), Format.formatDouble(item.getSDevResult(), 4),
                        Format.formatTimeSpan(item.getMeanElapsed())});
            }


        }
        holder.sort(new int[]{0, 1, 2});
        holder.write(file);

    }
}
