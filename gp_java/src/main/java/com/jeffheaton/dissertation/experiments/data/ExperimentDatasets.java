package com.jeffheaton.dissertation.experiments.data;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.experiments.payloads.PayloadGeneticFit;
import com.jeffheaton.dissertation.util.ObtainFallbackStream;
import com.jeffheaton.dissertation.util.ObtainInputStream;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.EncogError;
import org.encog.util.csv.CSVFormat;
import org.encog.util.csv.ReadCSV;

import java.util.*;

/**
 * Created by jeff on 7/4/16.
 */
public class ExperimentDatasets {

    private List<DatasetInfo> datasets = new ArrayList<>();
    private Map<String, DataCacheElement> cache = new HashMap<>();

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
        quick.clearUniques();
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

    public synchronized DataCacheElement loadDatasetNeural(ExperimentTask task) {
        String key = "neural:"+task.getDatasetFilename()+task.getModelType().getTarget()+task.getPredictors();

        if( this.cache.containsKey(key)) {
            //System.out.println("Neural Key Found: " + key);
            return this.cache.get(key);
        }

        //System.out.println("Neural Loading key: " + key);

        ObtainInputStream source = new ObtainFallbackStream(task.getDatasetFilename());
        QuickEncodeDataset quick = new QuickEncodeDataset(false,false);
        quick.analyze(source, task.getModelType().getTarget(), true, CSVFormat.EG_FORMAT);

        if( task.getPredictors()!=null && task.getPredictors().length()>0 ) {
            quick.forcePredictors(task.getPredictors());
        }

        if( !task.getModelType().isRegression() ) {
            quick.getTargetField().setEncodeType(QuickEncodeDataset.QuickFieldEncode.OneHot);
        }

        quick.clearUniques();
        DataCacheElement element = new DataCacheElement(quick);
        this.cache.put(key,element);
        return element;
    }

    public synchronized DataCacheElement loadDatasetGP(ExperimentTask task) {
        String key = "gp:"+task.getDatasetFilename()+task.getModelType().getTarget()+task.getPredictors();

        if( this.cache.containsKey(key)) {
            //System.out.println("GP Key Found: " + key);
            return this.cache.get(key);
        }

        //System.out.println("GP Loading key: " + key);

        ObtainInputStream source = new ObtainFallbackStream(task.getDatasetFilename());
        QuickEncodeDataset quick = new QuickEncodeDataset(true,false);
        quick.analyze(source, task.getModelType().getTarget(), true, CSVFormat.EG_FORMAT);

        if( task.getPredictors()!=null && task.getPredictors().length()>0 ) {
            quick.forcePredictors(task.getPredictors());
        }

        if( quick.getTargetField().getEncodeType()== QuickEncodeDataset.QuickFieldEncode.NumericCategory
                && quick.getTargetField().getUnique()>2) {
            throw new EncogError(PayloadGeneticFit.GP_CLASS_ERROR);
        }

        quick.clearUniques();
        DataCacheElement element = new DataCacheElement(quick);
        this.cache.put(key,element);
        return element;
    }
}
