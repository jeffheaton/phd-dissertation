package com.jeffheaton.dissertation.experiments.report;

import au.com.bytecode.opencsv.CSVWriter;
import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import org.encog.EncogError;
import org.encog.util.Format;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ReportHolder {
    private List<String[]> data = new ArrayList<>();
    private String[] headers;

    class RowComparator implements Comparator<String[]> {
        private int[] order;

        public RowComparator(int[] theOrder) {
            this.order = theOrder;
        }

        @Override
        public int compare(String[] line1, String[] line2) {
            for(int i=0;i<line1.length;i++) {
                int c = line1[i].compareTo(line2[i]);
                if( c!=0 ) {
                    return c;
                }
            }
            return 0;
        }
    }

    public void addLine(String[] line) {
        this.data.add(line);
    }

    public void setHeaders(String[] theHeaders) {
        this.headers = theHeaders;
    }

    public void sort(int[] order) {
        RowComparator comp = new RowComparator(order);
        Collections.sort(this.data,comp);
    }

    public void write(File file) {

        try (CSVWriter writer = new CSVWriter(new FileWriter(file))) {
            writer.writeNext(this.headers);
            writer.writeAll(this.data);
        } catch (IOException ex) {
            throw new EncogError(ex);
        }
    }
}
