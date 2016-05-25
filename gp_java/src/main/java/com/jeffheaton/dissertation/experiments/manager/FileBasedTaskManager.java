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
import java.util.Arrays;
import java.util.List;

import static java.nio.file.StandardOpenOption.*;

/**
 * Created by jeff on 5/16/16.
 */
public class FileBasedTaskManager implements TaskQueueManager {

    public final static String FILE_LOCK = "lock.lck";
    public final static String FILE_WORKLOAD = "workload.csv";

    private final Path path;
    private final Path pathLock;
    private final Path pathWorkload;
    private int maxWaitLock;

    public FileBasedTaskManager(File theFolder) {
        this.path = theFolder.toPath();
        this.pathLock = new File(theFolder, FILE_LOCK).toPath();
        this.pathWorkload = new File(theFolder, FILE_WORKLOAD).toPath();
    }

    private List<ExperimentTask> loadTasks() {

        List<ExperimentTask> result = new ArrayList<>();

        if( this.pathWorkload.toFile().exists() ) {
            try (CSVReader reader = new CSVReader(new FileReader(this.pathWorkload.toFile()));) {
                reader.readNext();
                String[] line;
                while ((line = reader.readNext()) != null) {
                    result.add(new ExperimentTask(line[0], line[1], line[2], Integer.parseInt(line[3])));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    private void saveTasks(List<ExperimentTask> tasks) {

        try (CSVWriter writer = new CSVWriter(new FileWriter(this.pathWorkload.toFile()));) {
            writer.writeNext(new String[] {"name","dataset","algorithm","cycle"});
            for(ExperimentTask task : tasks) {
                writer.writeNext(new String[] {task.getName(),task.getDataset(),task.getAlgorithm(),""+task.getCycle()});
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public String addTask(String name, String dataset, String algorithm, int cycle) {
        List<ExperimentTask> current = loadTasks();
        current.add(new ExperimentTask(name,dataset,algorithm,cycle));
        saveTasks(current);
        return null;
    }

    @Override
    public void remoteTask(String key) {

    }

    @Override
    public void removeAll() {

    }

    @Override
    public ExperimentTask requestTask() {
        return null;
    }

    @Override
    public void addTaskCycles(String name, String dataset, String algorithm, int cycles) {
        for(int i=0;i<cycles;i++) {
            addTask(name,dataset,algorithm,i+1);
        }
    }

    public int getMaxWaitLock() {
        return maxWaitLock;
    }

    public void setMaxWaitLock(int maxWaitLock) {
        this.maxWaitLock = maxWaitLock;
    }

    private void obtainLock(int maxWaitSeconds) {
        int tries = 0;
        while(tries<maxWaitSeconds) {
            try {
                Files.createFile(this.pathLock);
            } catch (FileAlreadyExistsException ex) {
                try {
                    Thread.sleep(1);
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
            Files.delete(this.pathWorkload);
        } catch (IOException ex) {
            throw new EncogError(ex);
        }
    }
}
