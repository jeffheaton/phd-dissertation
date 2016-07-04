package com.jeffheaton.dissertation.experiments.data;

import com.jeffheaton.dissertation.util.ObtainFallbackStream;
import com.jeffheaton.dissertation.util.ObtainInputStream;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by jeff on 7/4/16.
 */
public class ExperimentDatasets {

    private List<DatasetInfo> datasets = new ArrayList<>();

    private ExperimentDatasets() {
        ObtainInputStream source = new ObtainFallbackStream("datasets.csv");
        ReadCSV reader = new ReadCSV(source.obtain(),true,CSVFormat.DECIMAL_POINT);
        while(reader.next()) {
            boolean regression = reader.get(0).trim().equalsIgnoreCase("r");
            String name = reader.get(1).trim();
            String target = reader.get(2).trim();
            List<String> predictors = Arrays.asList(reader.get(3).trim().split("\\s*,\\s*"));
            List<String> experiments = Arrays.asList(reader.get(4).trim().split("\\s*,\\s*"));
            int targetElements = determineTargetElements(name,target);
            DatasetInfo info = new DatasetInfo(regression,name,target,predictors,experiments, targetElements);
            datasets.add(info);
        }
    }

    private static ExperimentDatasets instance;

    public int determineTargetElements(String dataset, String target) {
        ObtainInputStream source = new ObtainFallbackStream(dataset);
        QuickEncodeDataset quick = new QuickEncodeDataset(true,false);
        quick.analyze(source, target, true, CSVFormat.EG_FORMAT);
        return quick.getTargetField().getUnique();
    }

    public static ExperimentDatasets getInstance() {
        if( ExperimentDatasets.instance==null ) {
            ExperimentDatasets.instance = new ExperimentDatasets();
        }
        return ExperimentDatasets.instance;
    }

    public List<DatasetInfo> getDatasets() {
        return datasets;
    }

    public List<DatasetInfo> getDatasetsForExperiment(String name) {
        List<DatasetInfo> result = new ArrayList<>();
        for(DatasetInfo info: this.datasets) {
            if( info.getExperiments().contains(name)) {
                result.add(info);
            }
        }
        return result;
    }
}
