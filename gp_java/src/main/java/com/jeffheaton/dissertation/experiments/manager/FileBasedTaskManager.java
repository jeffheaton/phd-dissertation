package com.jeffheaton.dissertation.experiments.manager;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import org.encog.EncogError;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.AccessDeniedException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by jeff on 5/16/16.
 */
public class FileBasedTaskManager implements TaskQueueManager {

    public final static String FILE_LOCK = "lock.lck";
    public final static String FILE_WORKLOAD = "workload.csv";

    private final Path path;
    private final Path pathLock;
    private final Path pathWorkload;
    private final String computerName;

    public FileBasedTaskManager(File thePath) {
        this.path = thePath.toPath();
        this.pathLock = new File(thePath, FILE_LOCK).toPath();
        this.pathWorkload = new File(thePath, FILE_WORKLOAD).toPath();
        this.computerName = DissertationConfig.getInstance().getHost();
    }

    private List<ExperimentTask> loadTasks() {

        List<ExperimentTask> result = new ArrayList<>();

        if (Files.exists(this.pathWorkload)) {
            try (CSVReader reader = new CSVReader(new FileReader(this.pathWorkload.toFile()));) {
                reader.readNext();
                String[] line;
                while ((line = reader.readNext()) != null) {
                    ExperimentTask task = new ExperimentTask(line[0], line[2], line[3], line[4], Integer.parseInt(line[5]));
                    task.setResult(Double.parseDouble(line[6]));
                    task.setIterations(Integer.parseInt(line[7]));
                    task.setElapsed(Integer.parseInt(line[8]));
                    task.setStatus(line[1]);
                    task.setInfo(line[9]);
                    result.add(task);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    private void saveTasks(List<ExperimentTask> tasks) {

        try (CSVWriter writer = new CSVWriter(new FileWriter(this.pathWorkload.toFile()));) {
            writer.writeNext(new String[]{"name", "status", "dataset", "algorithm", "predictors", "cycle", "result", "iterations", "elapsed" });
            for (ExperimentTask task : tasks) {
                writer.writeNext(new String[]{task.getName(), task.getStatus(), task.getDatasetFilename(), task.getAlgorithm(),
                        task.getPredictors(),"" + task.getCycle(), "" + task.getResult(), "" + task.getIterations(),
                        "" + task.getElapsed(), task.getInfo()});
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private boolean taskExists(List<ExperimentTask> tasks, ExperimentTask searchTask) {
        for (ExperimentTask task : tasks) {
            if (task.getKey().equals(searchTask.getKey())) {
                return true;
            }
        }
        return false;
    }

    @Override
    public ExperimentTask addTask(String name, String dataset, String model, String predictors, int cycle) {
        List<ExperimentTask> currentTasks = loadTasks();
        ExperimentTask newTask = new ExperimentTask(name, dataset, model, predictors, cycle);
        if (taskExists(currentTasks, newTask)) {
            throw new EncogError("Task " + newTask.getKey() + " was already defined.");
        }
        currentTasks.add(newTask);
        saveTasks(currentTasks);
        return newTask;
    }

    @Override
    public void removeTask(String key) {
        List<ExperimentTask> currentTasks = loadTasks();
        Object[] currentTasksArray = currentTasks.toArray();
        for (int i = 0; i < currentTasksArray.length; i++) {
            if (((ExperimentTask) currentTasksArray[i]).getKey() == key) {
                currentTasks.remove(i);
                break;
            }
        }
        saveTasks(currentTasks);
    }

    @Override
    public void removeAll() {
        try {
            if (Files.exists(this.pathWorkload)) {
                Files.delete(this.pathWorkload);
            }
        } catch (IOException ex) {
            throw new EncogError(ex);
        }
        releaseLock();
    }

    @Override
    public ExperimentTask requestTask(int maxWaitSeconds) {
        try {
            obtainLock(maxWaitSeconds);

            List<ExperimentTask> currentTasks = loadTasks();
            for (ExperimentTask task : currentTasks) {
                if (task.isQueued()) {
                    task.claim(this.computerName);
                    saveTasks(currentTasks);
                    return task;
                }
            }

        } finally {
            releaseLock();
        }
        return null;
    }

    @Override
    public void addTaskCycles(String name, String dataset, String model, String predictors, int cycles) {
        for (int i = 0; i < cycles; i++) {
            addTask(name, dataset, model, predictors, i + 1);
        }
    }

    @Override
    public void reportDone(ExperimentTask task, int maxWaitSeconds) {
        try {
            obtainLock(maxWaitSeconds);

            List<ExperimentTask> currentTasks = loadTasks();
            Object[] currentTasksArray = currentTasks.toArray();
            for (int i = 0; i < currentTasksArray.length; i++) {
                if (((ExperimentTask) currentTasksArray[i]).getKey().equals(task.getKey())) {
                    task.reportDone(this.computerName);
                    currentTasks.set(i, task);
                    break;
                }
            }
            saveTasks(currentTasks);
        } finally {
            releaseLock();
        }
    }

    @Override
    public void blockUntilDone(int maxWaitSeconds) {
        for (; ; ) {
            try {
                obtainLock(maxWaitSeconds);
                boolean done = true;
                List<ExperimentTask> currentTasks = loadTasks();
                for (ExperimentTask currentTask : currentTasks) {
                    if (!currentTask.isComplete()) {
                        done = false;
                    }
                }
                if (done) {
                    return;
                }
            } finally {
                releaseLock();
            }
            try {
                Thread.sleep(2000);
            } catch (InterruptedException ex) {
                throw new EncogError(ex);
            }
        }
    }

    @Override
    public void reportError(ExperimentTask task, Exception ex, int maxWaitSeconds) {
        try {
            obtainLock(maxWaitSeconds);

            List<ExperimentTask> currentTasks = loadTasks();
            for (ExperimentTask currentTask : currentTasks) {
                if (currentTask.getKey().equals(task.getKey())) {
                    currentTask.reportError(this.computerName, ex);
                    saveTasks(currentTasks);
                    return;
                }
            }

        } finally {
            releaseLock();
        }
    }

    @Override
    public List<ExperimentTask> getQueue(int maxWaitSeconds) {
        try {
            obtainLock(maxWaitSeconds);
            return loadTasks();
        } finally {
            releaseLock();
        }
    }

    private void obtainLock(int maxWaitSeconds) {
        int tries = 0;
        while (tries < maxWaitSeconds) {
            try {
                Files.createFile(this.pathLock);
                return;
            } catch (FileAlreadyExistsException ex) {
                try {
                    Thread.sleep(1000);
                    tries++;
                } catch (InterruptedException ex2) {
                    throw new EncogError(ex2);
                }
            } catch(AccessDeniedException ex) {
                try {
                    Thread.sleep(1000);
                    tries++;
                } catch (InterruptedException ex2) {
                    throw new EncogError(ex2);
                }
            } catch (IOException ex) {
                ex.printStackTrace();
                throw new EncogError(ex);
            }
        }
        EncogError ex = new EncogError("Could not lock " + this.pathWorkload + " after " + maxWaitSeconds + ".");
        ex.printStackTrace();
        throw ex;
    }

    private void releaseLock() {
        try {
            if (Files.exists(this.pathLock)) {
                Files.delete(this.pathLock);
            }
        } catch (IOException ex) {
            ex.printStackTrace();
            throw new EncogError(ex);
        }
    }
}
