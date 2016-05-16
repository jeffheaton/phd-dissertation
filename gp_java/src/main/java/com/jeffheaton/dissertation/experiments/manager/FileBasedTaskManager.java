package com.jeffheaton.dissertation.experiments.manager;

import org.encog.EncogError;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;

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

    @Override
    public String addTask(String name, String dataset, String algorithm, int cycle) {
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
