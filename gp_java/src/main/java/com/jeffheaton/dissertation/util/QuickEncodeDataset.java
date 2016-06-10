package com.jeffheaton.dissertation.util;

import org.encog.EncogError;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.util.Format;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.io.File;
import java.io.InputStream;
import java.util.*;

/**
 * Created by jeff on 4/11/16.
 */
public class QuickEncodeDataset {

    public enum QuickFieldEncode {
        NumericCategory,
        OneHot,
        Ignore,
        RawNumeric
    }

    public static class QuickField {

        private final QuickEncodeDataset owner;
        private final String name;
        private final int index;
        private double min;
        private double max;
        private double mean;
        private double sd;
        private double sum;
        private boolean numeric;
        private boolean integer;
        private boolean missing;
        private int count;
        private Map<String,Integer> uniqueCounts = new HashMap<>();
        private Map<String,Integer> uniqueIndex = new HashMap<>();
        private QuickFieldEncode encodeType = QuickFieldEncode.Ignore;



        public static boolean isNA(String str) {
            if (str == null) {
                return true;
            }
            String t = str.trim();
            if (t.equals("") || t.equalsIgnoreCase("NA") || t.equals("?") || t.equalsIgnoreCase("null")) {
                return true;
            }
            return false;
        }

        public QuickField(QuickEncodeDataset theOwner, String theName, int theIndex) {
            this.owner = theOwner;
            this.name = theName;
            this.index = theIndex;
            this.numeric = true;
            this.integer = true;
            this.min = this.max = Double.NaN;
            this.mean = Double.NaN;
        }

        private void updateUnique(String str) {
            if(this.uniqueCounts.containsKey(str)) {
                this.uniqueCounts.put(str,this.uniqueCounts.get(str)+1);
            } else {
                this.uniqueCounts.put(str,1);
                this.uniqueIndex.put(str,this.uniqueIndex.size());
            }
        }

        public void updatePass1(String str) {
            updateUnique(str);

            if (isNA(str)) {
                this.missing = true;
                return;
            }

            if (this.integer) {
                try {
                    int intValue = Integer.parseInt(str);
                    if( Double.isNaN(this.max)) {
                        this.max = this.min = intValue;
                    } else {
                        this.max = Math.max(intValue, this.max);
                        this.min = Math.min(intValue, this.min);
                    }
                    this.sum += intValue;
                    this.count++;
                    return;
                } catch (NumberFormatException ex) {
                    this.integer = false;
                }
            }

            if (this.numeric) {
                try {
                    double floatValue = Double.parseDouble(str);
                    if( Double.isNaN(this.max)) {
                        this.max = this.min = floatValue;
                    } else {
                        this.max = Math.max(floatValue, this.max);
                        this.min = Math.min(floatValue, this.min);
                    }
                    this.sum += floatValue;
                    this.count++;
                    return;
                } catch (NumberFormatException ex) {
                    this.numeric = false;
                }
            }

            this.count++;
        }

        public void finalizePass1() {
            if (this.count > 0 && this.numeric) {
                this.mean = this.sum / this.count;
                this.sum = 0;
            }
        }

        public void updatePass2(String str) {
            if (this.count > 0 && !isNA(str) && this.numeric) {
                double d = this.mean - Double.parseDouble(str);
                this.sum += d * d;
            }
        }

        public boolean isLowRange() {
            if(!isInteger()) {
                return false;
            }
            if( ((int)getMin()!=0) && ((int)getMin()!=1) )
                return false;

            return ((int)getMax()-(int)getMin())<=5;
        }

        public void finalizePass2() {
            if (this.count > 0 && this.numeric) {
                this.sd = Math.sqrt(this.sum / this.count);
            }

            if( !this.isNumeric() && this.getUnique()>100 ) {
                // Pure string field, can't use it
                this.encodeType = QuickFieldEncode.Ignore;
            } else if( this.calculateMissingPercent()>0.5 ) {
                // Too many missing, can't use it
                this.encodeType = QuickFieldEncode.Ignore;
            } else if( this.isInteger() && this.calculateUniquePercent()==1.0 ) {
                // Likely an ID field
                this.encodeType = QuickFieldEncode.Ignore;
            } else if (isLowRange()) {
                this.encodeType = QuickFieldEncode.OneHot;
            } else if( !this.isNumeric() ) {
                // Treat as catagorical
                this.encodeType = QuickFieldEncode.OneHot;
            } else if( this.numeric ) {
                this.encodeType = QuickFieldEncode.RawNumeric;
            } else {
                // I don't think it reaches this point, but ignore just in case...
                this.encodeType = QuickFieldEncode.Ignore;
            }
        }

        public String getName() {
            return name;
        }

        public boolean hasMissing() {
            return this.missing;
        }

        public int getIndex() {
            return this.index;
        }

        public double getMin() {
            return min;
        }

        public void setMin(double min) {
            this.min = min;
        }

        public double getMax() {
            return max;
        }

        public void setMax(double max) {
            this.max = max;
        }

        public double getMean() {
            return mean;
        }

        public void setMean(double mean) {
            this.mean = mean;
        }

        public double getSd() {
            return sd;
        }

        public void setSd(double sd) {
            this.sd = sd;
        }

        public boolean isNumeric() {
            return numeric;
        }

        public int getCount() {
            return count;
        }

        public void setCount(int count) {
            this.count = count;
        }

        public boolean isInteger() {
            return this.integer;
        }

        public boolean isMissing() {
            return missing;
        }

        public void setMissing(boolean missing) {
            this.missing = missing;
        }

        public int getUnique() {
            return this.uniqueCounts.size();
        }

        public QuickFieldEncode getEncodeType() {
            return encodeType;
        }

        public void setEncodeType(QuickFieldEncode encodeType) {
            this.encodeType = encodeType;
        }

        public double calculateUniquePercent() {
            return (double)this.getUnique()/(double)this.getCount();
        }

        public double calculateMissingPercent() {
            return (double)this.getMissing()/(double)this.owner.getCount();
        }

        public int getMissing() {
            return this.owner.getCount() - this.getCount();
        }

        public int encodeColumnsNeeded() {
            switch(getEncodeType()) {
                case Ignore:
                    return 0;
                case RawNumeric:
                    return 1;
                case NumericCategory:
                    return 1;
                case OneHot:
                    return getUnique();
                default:
                    // Should not happen.
                    return 0;
            }
        }

        public int encode(int startIndex, String str, double[] vec) {
            switch(getEncodeType()) {
                case Ignore:
                    return startIndex;
                case RawNumeric:
                    if(isNA(str)) {
                        vec[startIndex] = this.mean;
                    } else {
                        vec[startIndex] = Double.parseDouble(str);
                    }
                    return startIndex+1;
                case NumericCategory:
                    vec[startIndex] = findCategoryIndex(str);
                    return startIndex+1;
                case OneHot:
                    int len = encodeColumnsNeeded();
                    int idx = findCategoryIndex(str);
                    for(int i=0;i<len;i++) {
                        if(i==idx) {
                            vec[startIndex+i] = 1;
                        } else {
                            vec[startIndex + i] = 0;
                        }
                    }
                    return startIndex+getUnique();
                default:
                    // Should not happen.
                    return startIndex;
            }
        }

        public int findCategoryIndex(String str) {
            return this.uniqueIndex.get(str);
        }

        @Override
        public String toString() {
            StringBuilder result = new StringBuilder();
            result.append("[QuickField: name=");
            result.append(getName());
            result.append(",mean=");
            result.append(getMean());
            result.append(",sd=");
            result.append(getSd());
            result.append(",min=");
            result.append(getMin());
            result.append(",max=");
            result.append(getMax());
            result.append(",type=");
            if( isInteger() ) {
                result.append("integer");
            } else if( isNumeric() ) {
                result.append("numeric");
            } else {
                result.append("string");
            }
            result.append(",unique=");
            result.append(getUnique());
            result.append("("+ Format.formatPercent(calculateUniquePercent())+")");
            result.append(",missing=");
            result.append(getMissing());
            result.append("("+ Format.formatPercent(calculateMissingPercent())+")");
            result.append(",encodeType=");
            result.append(getEncodeType());
            result.append("]");
            return result.toString();
        }
    }

    private boolean headers;
    private CSVFormat format;
    private ObtainInputStream streamSource;
    private QuickField targetField;
    private final List<QuickField> predictors = new ArrayList<>();
    private final List<QuickField> fields = new ArrayList<>();
    private int count;


    private void processPass1(String theTargetColumn) {
        InputStream stream = this.streamSource.obtain();
        ReadCSV csv = new ReadCSV(stream, headers, format);

        for (int i = 0; i < csv.getColumnNames().size(); i++) {
            QuickField field = new QuickField(this,csv.getColumnNames().get(i),i);
            this.fields.add(field);
            if( field.getName().equals(theTargetColumn)) {
                this.targetField = field;
            }
        }

        if( this.targetField==null ) {
            throw new EncogError("Can't find target column: " + theTargetColumn);
        }

        this.count = 0;
        while (csv.next()) {
            for (QuickField field : this.fields) {
                field.updatePass1(csv.get(field.getIndex()));
            }
            this.count++;
        }
        csv.close();


        for (QuickField field : this.fields) {
            field.finalizePass1();
        }
    }

    private void processPass2() {
        InputStream stream = this.streamSource.obtain();
        ReadCSV csv = new ReadCSV(stream, headers, format);

        while (csv.next()) {
            for (QuickField field : this.fields) {
                field.updatePass2(csv.get(field.getIndex()));
            }
        }
        csv.close();

    }

    public int getCount() {
        return this.count;
    }

    private boolean isin(String b, String[] a) {
        for (int i = 0; i < a.length; i++) {
            if (b.equals(a[i])) {
                return true;
            }
        }
        return false;
    }

    public int encodedColumnsNeededX() {
        int result = 0;
        for(QuickField field: this.predictors) {
            result+=field.encodeColumnsNeeded();
        }
        return result;
    }

    public int encodedColumnsNeededY() {
        return this.targetField.encodeColumnsNeeded();
    }

    private MLDataSet processPass3() {
        for (QuickField field : this.fields) {
            field.finalizePass2();
            if( field!=this.targetField && field.getEncodeType()!=QuickFieldEncode.Ignore) {
                this.predictors.add(field);
            }
        }

        InputStream stream = this.streamSource.obtain();
        ReadCSV csv = new ReadCSV(stream, headers, format);
        BasicMLDataSet result = new BasicMLDataSet();
        double[] xVector = new double[encodedColumnsNeededX()];
        double[] yVector = new double[encodedColumnsNeededY()];

        while(csv.next()) {
            int idx = 0;
            for(QuickField field: this.predictors) {
                idx = field.encode(idx,csv.get(field.getIndex()),xVector);
            }
            this.targetField.encode(0,csv.get(this.targetField.getIndex()),yVector);
            MLData xData = new BasicMLData(xVector);
            MLData yData = new BasicMLData(yVector);
            result.add(xData,yData);
        }

        csv.close();
        return result;
    }

    public void dumpFieldInfo() {
        for (QuickField field : this.fields) {
            System.out.println(field.toString());
        }
    }


    public MLDataSet process(ObtainInputStream theStreamSource, String theTargetColumn, String thePredictorColumns,
                             boolean theHeaders, CSVFormat theFormat) {
        this.streamSource = theStreamSource;
        this.headers = theHeaders;
        this.format = theFormat;

        processPass1(theTargetColumn);
        processPass2();

        if( thePredictorColumns !=null ) {
            // we've been provided a list of predictors to use, ignore others
            List<String> used = Arrays.asList(thePredictorColumns.split(","));
            for(QuickField field: this.fields) {
                if( field!=this.targetField && !used.contains(field.getName()) ) {
                    field.setEncodeType(QuickFieldEncode.Ignore);
                }
            }
        }



        return processPass3();
    }

    public List<QuickField> getPredictors() {
        return this.predictors;
    }

    public List<QuickField> getFields() {
        return this.fields;
    }

    public static void main(String[] args) {
        ObtainInputStream source = new ObtainFallbackStream("auto-mpg.csv");
        QuickEncodeDataset quick = new QuickEncodeDataset();
        MLDataSet dataset = quick.process(source,"mpg", null, true, CSVFormat.EG_FORMAT);
        quick.dumpFieldInfo();
        System.out.println(quick.getCount());
    }
}
