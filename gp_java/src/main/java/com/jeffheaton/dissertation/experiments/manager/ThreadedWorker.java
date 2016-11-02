package com.jeffheaton.dissertation.experiments.manager;

public class ThreadedWorker implements Runnable {

    private ThreadedRunner runner;
    private boolean requestShutdown = false;

    public ThreadedWorker(ThreadedRunner theRunner) {
        this.runner = theRunner;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void run() {
        try {
            this.requestShutdown = false;
            while (!this.requestShutdown) {
                ExperimentTask task = this.runner.getManager().requestTask(runner.getMaxWait());

                if (task != null) {
                    task.setOwner(this.runner);
                    task.clearLog();
                    try {
                        task.run();
                        this.runner.getManager().reportDone(task, runner.getMaxWait());
                    } catch (Throwable ex) {
                        ex.printStackTrace();
                        this.runner.getManager().reportError(task, ex, runner.getMaxWait());
                    }
                } else {
                    try {
                        Thread.sleep(5000);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }

            }
        } catch(Throwable ex) {
            ex.printStackTrace();
        }
    }

    public void shutdown() {
        this.requestShutdown = true;
    }
}
