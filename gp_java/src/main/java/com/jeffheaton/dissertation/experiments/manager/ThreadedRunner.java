package com.jeffheaton.dissertation.experiments.manager;

import org.encog.ml.data.MLDataSet;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by jeff on 5/16/16.
 */
public class ThreadedRunner {
    private final TaskQueueManager manager;
    private ExecutorService threadPool;
    private final List<ThreadedWorker> workers = new ArrayList<>();
    private int maxWait = 10;
    private boolean verbose;

    public ThreadedRunner(TaskQueueManager theManager) {
        this.manager = theManager;
    }

    public TaskQueueManager getManager() {
        return manager;
    }

    public void startup() {
        int processors = Runtime.getRuntime().availableProcessors();
        this.threadPool = Executors.newFixedThreadPool(processors);
        for(int i=0;i<processors;i++) {
            ThreadedWorker worker = new ThreadedWorker(this);
            this.workers.add(worker);
            this.threadPool.submit(worker);
        }
    }

    public int getMaxWait() {
        return maxWait;
    }

    public void shutdown() {
        for(ThreadedWorker worker: this.workers) {
            worker.shutdown();
        }
        this.workers.clear();
        this.threadPool.shutdown();
    }

    public boolean isVerbose() {
        return verbose;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }
}
