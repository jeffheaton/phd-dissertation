package com.jeffheaton.dissertation.experiments.report;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import org.encog.util.Format;

import java.io.File;
import java.util.List;

public class GenerateSimpleReport {
    private TaskQueueManager manager;
    private String i1Name;
    private String i2Name;

    public GenerateSimpleReport(TaskQueueManager theManager) {
        this.manager = theManager;
    }

    public void report(File file, String name, int max) {
        List<ExperimentTask> queue = this.manager.getQueue(max);

        ReportHolder holder = new ReportHolder();
        if( this.i1Name != null ) {
            holder.setHeaders(new String[]{"experiment", "algorithm", "dataset", "result", this.i1Name, this.i2Name, "elapsed" });
        } else {
            holder.setHeaders(new String[]{"experiment", "algorithm", "dataset", "result", "elapsed" });
        }


        for( ExperimentTask task: queue) {
            if( task.getName().equals(name) ) {
                if( this.i1Name != null ) {
                    holder.addLine(new String[]{task.getName(), task.getAlgorithm(),
                            task.getDatasetFilename(),
                            task.getInfo(),
                            "" + task.getI1(),
                            "" + task.getI2(),
                            Format.formatTimeSpan(task.getElapsed())});
                } else {
                    holder.addLine(new String[]{task.getName(), task.getAlgorithm(), task.getDatasetFilename(), task.getInfo(), Format.formatTimeSpan(task.getElapsed())});
                }

            }
        }
        holder.sort(new int[]{0, 1, 2});
        holder.write(file);
    }

    public void setIFieldNames(String theI1, String theI2) {
        this.i1Name = theI1;
        this.i2Name = theI2;
    }
}
