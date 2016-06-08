package com.jeffheaton.dissertation.experiments.data;

import com.jeffheaton.dissertation.util.ObtainFallbackStream;
import com.jeffheaton.dissertation.util.ObtainInputStream;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by jeff on 6/7/16.
 */
public class AnalyzeEngineeredDataset {

    public static final String ENGINEERED_FILENAME = "feature_eng.csv";

    private final Map<String,Integer> map = new HashMap<>();
    private final List<String> features = new ArrayList<>();

    public AnalyzeEngineeredDataset() {
        String regexp = "([A-Za-z0-9_]+)-([A-Za-z])([0-9])";

        Pattern pattern = Pattern.compile(regexp);


        ObtainInputStream source = new ObtainFallbackStream(ENGINEERED_FILENAME);
        ReadCSV csv = new ReadCSV(source.obtain(),true, CSVFormat.ENGLISH);
        for(String str: csv.getColumnNames()) {
            Matcher matcher = pattern.matcher(str);
            if(matcher.find()) {
                char var = matcher.group(2).charAt(0);

                if( var=='x' ) {
                    int n = Integer.parseInt(matcher.group(3));
                    String name = matcher.group(1);
                    if(this.map.containsKey(name)) {
                        int i = this.map.get(name);
                        if(n>i) {
                            this.map.put(name,n);
                        }
                    } else {
                        this.map.put(name,n);
                    }
                }
            }
        }

        for(String key: this.map.keySet()) {
            this.map.put(key,this.map.get(key)+1);
        }

        this.features.addAll(this.map.keySet());
    }

    public List<String> getFeatures() {
        return this.features;
    }

    public int getFeaturePredictorCount(String feature) {
        return this.map.get(feature);
    }

    public String getPredictors(String f) {
        StringBuilder result = new StringBuilder();
        int count = getFeaturePredictorCount(f);
        for(int i=0;i<count;i++) {
            if(i>0) {
                result.append(',');
            }
            result.append(f);
            result.append("-x");
            result.append(i);
        }
        return result.toString();
    }
}
