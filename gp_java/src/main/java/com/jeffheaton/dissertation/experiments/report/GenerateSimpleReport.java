package com.jeffheaton.dissertation.experiments.report;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.experiments.manager.TaskQueueManager;
import org.encog.util.Format;

import java.io.File;
import java.util.List;

/**
 * Created by jeffh on 6/26/2016.
 */
public class GenerateSimpleReport {
    private TaskQueueManager manager;

    public GenerateSimpleReport(TaskQueueManager theManager) {
        this.manager = theManager;
    }

    public void report(File file, String name, int max) {
        List<ExperimentTask> queue = this.manager.getQueue(max);

        ReportHolder holder = new ReportHolder();
        holder.setHeaders(new String[]{"experiment", "algorithm", "dataset", "result", "elapsed" });

        for( ExperimentTask task: queue) {
            if( task.getName().equals(name) ) {
                holder.addLine(new String[]{task.getName(), task.getAlgorithm(), task.getDatasetFilename(), task.getInfo(), Format.formatTimeSpan(task.getElapsed())});
            }
        }
        holder.sort(new int[]{0, 1, 2});
        holder.write(file);

    }
}
