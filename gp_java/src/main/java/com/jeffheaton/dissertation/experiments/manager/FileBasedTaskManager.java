package com.jeffheaton.dissertation.experiments.manager;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import org.encog.EncogError;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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

    public FileBasedTaskManager(File theFolder, String theComputerName) {
        this.path = theFolder.toPath();
        this.pathLock = new File(theFolder, FILE_LOCK).toPath();
        this.pathWorkload = new File(theFolder, FILE_WORKLOAD).toPath();
        this.computerName = theComputerName;
    }

    private List<ExperimentTask> loadTasks() {

        List<ExperimentTask> result = new ArrayList<>();

        if( Files.exists(this.pathWorkload) ) {
            try (CSVReader reader = new CSVReader(new FileReader(this.pathWorkload.toFile()));) {
                reader.readNext();
                String[] line;
                while ((line = reader.readNext()) != null) {
                    ExperimentTask task = new ExperimentTask(line[0], line[2], line[3], Integer.parseInt(line[4]));
                    task.setStatus(line[1]);
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
            writer.writeNext(new String[] {"name","status","dataset","algorithm","cycle"});
            for(ExperimentTask task : tasks) {
                writer.writeNext(new String[] {task.getName(),task.getStatus(),task.getDataset(),task.getAlgorithm(),""+task.getCycle()});
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private boolean taskExists(List<ExperimentTask> tasks, ExperimentTask searchTask) {
        for(ExperimentTask task: tasks) {
            if(task.getKey().equals(searchTask.getKey())) {
                return true;
            }
        }
        return false;
    }

    @Override
    public ExperimentTask addTask(String name, String dataset, String algorithm, int cycle) {
        List<ExperimentTask> currentTasks = loadTasks();
        ExperimentTask newTask = new ExperimentTask(name,dataset,algorithm,cycle);
        if( taskExists(currentTasks,newTask)) {
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
        for(int i=0;i<currentTasksArray.length;i++) {
            if( ((ExperimentTask)currentTasksArray[i]).getKey()==key ) {
                currentTasks.remove(i);
                break;
            }
        }
        saveTasks(currentTasks);
    }

    @Override
    public void removeAll() {
        try {
            if(Files.exists(this.pathWorkload)) {
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
                if( task.isQueued() ) {
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
    public void addTaskCycles(String name, String dataset, String algorithm, int cycles) {
        for(int i=0;i<cycles;i++) {
            addTask(name,dataset,algorithm,i+1);
        }
    }

    @Override
    public void reportDone(ExperimentTask task, int maxWaitSeconds) {
        try {
            obtainLock(maxWaitSeconds);

            List<ExperimentTask> currentTasks = loadTasks();
            for (ExperimentTask currentTask : currentTasks) {
                if( currentTask.getKey().equals(task.getKey()) ) {
                    currentTask.reportDone(this.computerName);
                    saveTasks(currentTasks);
                    return;
                }
            }

        } finally {
            releaseLock();
        }
    }

    @Override
    public void blockUntilDone(int maxWaitSeconds) {
        for(;;) {
            try {
                obtainLock(maxWaitSeconds);
                List<ExperimentTask> currentTasks = loadTasks();
                for (ExperimentTask currentTask : currentTasks) {
                    if( currentTask.isComplete() ) {
                        return;
                    }
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

    private void obtainLock(int maxWaitSeconds) {
        int tries = 0;
        while(tries<maxWaitSeconds) {
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
            } catch (IOException ex) {
                throw new EncogError(ex);
            }
        }
        throw new EncogError("Could not lock " + this.pathWorkload + " after " + maxWaitSeconds + ".");
    }

    private void releaseLock() {
        try {
            if(Files.exists(this.pathLock)) {
                Files.delete(this.pathLock);
            }
        } catch (IOException ex) {
            throw new EncogError(ex);
        }
    }
}
