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
    private int maxWait = 600;
    private boolean verbose;
    private int currentCount;

    public ThreadedRunner(TaskQueueManager theManager) {
        this.manager = theManager;
        this.verbose = DissertationConfig.getInstance().getVerbose();
    }

    public TaskQueueManager getManager() {
        return manager;
    }

    public void startup() {
        int threads = DissertationConfig.getInstance().getThreads();
        System.out.println("Using " + threads + " threads.");

        this.threadPool = Executors.newFixedThreadPool(threads);
        for(int i=0;i<threads;i++) {
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
        if( DissertationConfig.getInstance().isVerboseForced() ) {
            this.verbose = verbose;
        }
    }

    public synchronized void reportComplete(ExperimentTask task) {
        int totalCount = this.getManager().getQueue(this.maxWait).size();
        this.currentCount++;
        System.out.println("Complete: " + this.currentCount + "/" + totalCount + ", " + task.getKey()
                + " - " + task.getResult() );

    }
}
