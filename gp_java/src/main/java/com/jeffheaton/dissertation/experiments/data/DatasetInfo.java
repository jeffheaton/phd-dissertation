package com.jeffheaton.dissertation.experiments.data;

import com.jeffheaton.dissertation.experiments.manager.ExperimentTask;
import com.jeffheaton.dissertation.experiments.payloads.PayloadGeneticFit;
import com.jeffheaton.dissertation.util.ObtainFallbackStream;
import com.jeffheaton.dissertation.util.ObtainInputStream;
import com.jeffheaton.dissertation.util.QuickEncodeDataset;
import org.encog.EncogError;
import org.encog.util.csv.CSVFormat;

import java.util.*;

/**
 * Created by jeff on 7/4/16.
 */
public class DatasetInfo {

    private final boolean regression;
    private final String name;
    private final String target;
    private final List<String> predictors = new ArrayList<>();
    private final List<String> experiments = new ArrayList<>();
    private final int targetElements;
    private Map<String, DataCacheElement> cache = new HashMap<>();

    public DatasetInfo(boolean theRegression, String theName, String theTarget,
                       List<String> thePredictors, List<String> theExperiments) {
        this.regression = theRegression;
        this.name = theName;
        this.target = theTarget;
        this.predictors.addAll(thePredictors);
        this.experiments.addAll(theExperiments);

        QuickEncodeDataset quick = loadDatasetNeural(target,thePredictors).getQuick();
        this.targetElements = quick.getTargetField().getUnique();
    }

    public boolean isRegression() {
        return regression;
    }

    public String getName() {
        return name;
    }

    public String getTarget() {
        return target;
    }

    public List<String> getPredictors() {
        return predictors;
    }

    public List<String> getExperiments() {
        return experiments;
    }

    public int getTargetElements() {
        return targetElements;
    }

    public synchronized DataCacheElement loadDatasetNeural(String target, List<String> predictors) {
        String key = "neural:"+target+":"+predictors;

        if( this.cache.containsKey(key)) {
            return this.cache.get(key);
        }

        ObtainInputStream source = new ObtainFallbackStream(this.name);
        QuickEncodeDataset quick = new QuickEncodeDataset(false,true);
        quick.analyze(source, target, true, CSVFormat.EG_FORMAT);

        if( predictors!=null && predictors.size()>0 ) {
            quick.forcePredictors(predictors);
        }

        if( !isRegression() ) {
            quick.getTargetField().setEncodeType(QuickEncodeDataset.QuickFieldEncode.OneHot);
        }

        quick.clearUniques();
        DataCacheElement element = new DataCacheElement(quick);
        this.cache.put(key,element);
        return element;
    }

    public synchronized DataCacheElement loadDatasetGP(String target, List<String> predictors) {
        String key = "gps:"+target+":"+predictors;

        if( this.cache.containsKey(key)) {
            return this.cache.get(key);
        }


        ObtainInputStream source = new ObtainFallbackStream(getName());
        QuickEncodeDataset quick = new QuickEncodeDataset(true,false);
        quick.analyze(source, target, true, CSVFormat.EG_FORMAT);

        if( predictors!=null && predictors.size()>0 ) {
            quick.forcePredictors(predictors);
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

    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("[Dataset:");
        result.append(this.name);
        result.append(",regression=");
        result.append(this.regression);
        result.append(",target=");
        result.append(this.target);
        result.append(",predictors=");
        result.append(this.predictors);
        result.append(",experiments=");
        result.append(this.experiments);
        result.append(",targetElements=");
        result.append(this.targetElements);
        result.append("]");
        return result.toString();
    }
}
