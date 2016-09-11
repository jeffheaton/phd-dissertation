package com.jeffheaton.dissertation.experiments.data;

import com.jeffheaton.dissertation.JeffDissertation;
import com.jeffheaton.dissertation.experiments.manager.DissertationConfig;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.persist.source.ObtainFallbackStream;
import org.encog.persist.source.ObtainInputStream;
import org.encog.util.EngineArray;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.util.*;

/**
 * Created by jeff on 7/4/16.
 */
public class ExperimentDatasets {

    private List<DatasetInfo> datasets = new ArrayList<>();

    private ExperimentDatasets() {
        ObtainInputStream source = new ObtainFallbackStream(DissertationConfig.getInstance().getDataPath().toString(),
                "datasets.csv", JeffDissertation.class);
        ReadCSV reader = new ReadCSV(source.obtain(), true, CSVFormat.DECIMAL_POINT);
        while (reader.next()) {
            boolean regression = reader.get(0).trim().equalsIgnoreCase("r");
            String name = reader.get(1).trim();
            String target = reader.get(2).trim();
            List<String> predictors = EngineArray.string2list(reader.get(3).trim());
            List<String> experiments = EngineArray.string2list(reader.get(4).trim());
            DatasetInfo info = new DatasetInfo(regression, name, target, predictors, experiments);
            datasets.add(info);
        }
    }

    private static ExperimentDatasets instance;

    public int determineTargetElements(String dataset, String target) {
        QuickEncodeDataset quick = this.loadDatasetNeural(dataset,target,null).getQuick();
        return quick.getTargetField().getUnique();
    }

    public static ExperimentDatasets getInstance() {
        if (ExperimentDatasets.instance == null) {
            ExperimentDatasets.instance = new ExperimentDatasets();
        }
        return ExperimentDatasets.instance;
    }

    public List<DatasetInfo> getDatasets() {
        return datasets;
    }

    public List<DatasetInfo> getDatasetsForExperiment(String name) {
        List<DatasetInfo> result = new ArrayList<>();
        for (DatasetInfo info : this.datasets) {
            if (info.getExperiments().contains(name)) {
                result.add(info);
            }
        }
        return result;
    }

    public DatasetInfo findDataset(String name) {
        for (DatasetInfo info : this.datasets) {
            if (name.equals(info.getName())) {
                return info;
            }
        }
        return null;
    }

    public DataCacheElement loadDatasetNeural(String name, String target, List<String> predictors) {
        DatasetInfo info = findDataset(name);
        return info.loadDatasetNeural(target,predictors);
    }

    public DataCacheElement loadDatasetGP(String name, String target, List<String> predictors) {
        DatasetInfo info = findDataset(name);
        return info.loadDatasetGP(target,predictors);
    }
}
