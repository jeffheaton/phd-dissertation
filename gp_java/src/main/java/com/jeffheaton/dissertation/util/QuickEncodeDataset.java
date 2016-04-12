package com.jeffheaton.dissertation.util;

import org.encog.ml.data.MLDataSet;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.File;

/**
 * Created by jeff on 4/11/16.
 */
public class QuickEncodeDataset {

    private double[] min;
    private double[] max;
    private double[] mean;
    private double[] sdev;
    private boolean[] numeric;
    private String[] name;
    private int[] rowCount;

    private File file;
    private boolean headers;
    private CSVFormat format;
    private int targetColumn;

    private boolean isNA(String str) {
        if(str.trim().equalsIgnoreCase("NA")|| str.trim().equals("?")) {
            return true;
        } else {
            return false;
        }
    }


    private void processPass1() {
        ReadCSV csv = new ReadCSV(file,headers,format);
        int count = 0;

        while(csv.next()) {
            if( this.min==null) {
                count = csv.getColumnCount();
                this.min = new double[count];
                this.max = new double[count];
                this.mean = new double[count];
                this.sdev = new double[count];
                this.numeric = new boolean[count];
                this.name = new String[count];
                this.rowCount = new int[count];
                for(int i=0;i<this.numeric.length;i++) {
                    this.numeric[i]=true;
                    this.min[i]=Double.POSITIVE_INFINITY;
                    if( this.headers ) {
                        this.name[i] = csv.getColumnNames().get(i);
                    } else {
                        this.name[i] = "field-"+i;
                    }
                }
            }

            for(int i=0;i<count;i++) {
                if(!this.numeric[i]) {
                    continue;
                }

                try {
                    if(!isNA(csv.get(i))) {
                        double d = Double.parseDouble(csv.get(i));
                        this.min[i] = Math.min(d, this.min[i]);
                        this.max[i] = Math.max(d, this.max[i]);
                        this.mean[i] += d;
                        this.rowCount[i]++;
                    }
                } catch (NumberFormatException ex) {
                    this.numeric[i] = false;
                }
            }
        }
        csv.close();
        for(int i=0;i<this.mean.length;i++) {
            if(this.numeric[i]) {
                this.mean[i] /= this.rowCount[i];
            }
        }
    }

    private void processPass2() {
        ReadCSV csv = new ReadCSV(file,headers,format);

        while(csv.next()) {
            for(int i=0;i<this.numeric.length;i++) {
                if(this.numeric[i]) {
                    String str = csv.get(i);
                    if( !isNA(str)) {
                        double d = Double.parseDouble(str)-this.mean[i];
                        this.sdev[i] += d*d;
                    }
                }
            }
        }
        csv.close();
        for(int i=0;i<this.mean.length;i++) {
            if(this.numeric[i]) {
                this.sdev[i] /= this.rowCount[i];
            }
        }
    }

    private MLDataSet processPass3() {
        int inputFeatureCount = 0;
        for(int i=0;i<this.numeric.length;i++) {
            if(this.numeric[i]) {
                inputFeatureCount++;
            }
        }
        int[] inputFeatures = new int[inputFeatureCount-1];
        int idx = 0;
        for(int i=0;i<this.numeric.length;i++) {
            if(this.numeric[i] && i!=this.targetColumn) {
                inputFeatures[idx++] = i;
            }
        }
        int[] targetFeatures = new int[1];

        ReadCSV csv = new ReadCSV(file,headers,format);
        MLDataSet result = Loader.loadCSV(csv,inputFeatures,targetFeatures);
        csv.close();
        return result;
    }

    public void dumpFieldInfo() {
        for(int i=0;i<this.name.length;i++) {
            if( this.numeric[i] ) {
                System.out.println(i + ":" + this.name[i] + ",min=" + this.min[i] + ",max=" + this.max[i] + ",mean=" + this.mean[i]+",sdev="+this.sdev[i]);
            } else {
                System.out.println(i + ":" + this.name[i] + "(ignored)");
            }
        }
    }


    public MLDataSet process(File theFile, int targetColumn, boolean theHeaders, CSVFormat theFormat) {
        this.file = theFile;
        this.headers = theHeaders;
        this.format = theFormat;
        processPass1();
        processPass2();
        return processPass3();
    }

    private boolean skip(int n) {
        return !this.numeric[n] || n==0 || n==1;
    }

    public double[] getMin() {
        return min;
    }

    public double[] getMax() {
        return max;
    }

    public double[] getMean() {
        return mean;
    }

    public double[] getSdev() {
        return sdev;
    }

    public boolean[] getNumeric() {
        return numeric;
    }

    public String[] getName() {
        return name;
    }

    public int[] getRowCount() {
        return rowCount;
    }

    public void analyze(File theFile, int targetColumn, boolean theHeaders, CSVFormat theFormat) {
        this.file = theFile;
        this.headers = theHeaders;
        this.format = theFormat;
        processPass1();
        processPass2();

        int idx = 0;
        for(int w=0;w<this.numeric.length;w++) {
            if( skip(w) )
                continue;

            for(int x=0;x<this.numeric.length;x++) {

                if( w==x  || skip(x) )
                    continue;

                for(int y=0;y<this.numeric.length;y++) {

                    if( y==w || y==x  || skip(y) )
                        continue;

                    for(int z=0;z<this.numeric.length;z++) {
                        if( z==y || z==w || z==x || skip(z) )
                            continue;

                        StringBuilder b = new StringBuilder();
                        b.append("(");
                        b.append(this.name[w]);
                        b.append("-");
                        b.append(this.name[x]);
                        b.append(")/");
                        b.append("(");
                        b.append(this.name[y]);
                        b.append("-");
                        b.append(this.name[z]);
                        b.append(")/");

                        System.out.println(idx + ":" + b.toString());
                        idx++;


                    }
                }
            }
        }
    }

    public int getTargetColumn() {
        return targetColumn;
    }
}
