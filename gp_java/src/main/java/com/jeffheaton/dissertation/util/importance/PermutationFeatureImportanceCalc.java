package com.jeffheaton.dissertation.util.importance;

import org.encog.EncogError;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.MLRegression;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.util.EngineArray;
import org.encog.util.simple.EncogUtility;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by jeff on 6/17/16.
 */
public class PermutationFeatureImportanceCalc extends AbstractFeatureImportance {

    private GenerateRandom rnd = new MersenneTwisterGenerateRandom();

    private MLDataSet generatePermutation(MLDataSet source, int column) {
        MLDataSet result = new BasicMLDataSet();
        for(MLDataPair item:source) {
            BasicMLData input = new BasicMLData(item.getInput().size());
            BasicMLData ideal = new BasicMLData(item.getIdeal().size());
            EngineArray.arrayCopy(item.getInputArray(),input.getData());
            EngineArray.arrayCopy(item.getIdealArray(),ideal.getData());
            MLDataPair newPair = new BasicMLDataPair(input,ideal);
            result.add(newPair);
        }

        for(int i=0;i<result.size();i++) {
            int r = i + rnd.nextInt(result.size()-i);
            MLData rowR = result.get(r).getInput();
            MLData rowI = result.get(i).getInput();

            double t = rowR.getData(column);
            rowR.setData(column,rowI.getData(column));
            rowI.setData(column,t);
        }

        return result;
    }

    @Override
    public void performRanking() {
        throw new EncogError("This algorithm requires a dataset to measure performance against, please call performRanking with a dataset.");
    }

    @Override
    public void performRanking(MLDataSet theDataset) {

        double baseline = EncogUtility.calculateRegressionError(getModel(),theDataset);

        double max = 0;
        for(int i=0;i<getModel().getInputCount();i++) {
            FeatureRank fr = getFeatures().get(i);
            MLDataSet p = generatePermutation(theDataset,i);
            double e = EncogUtility.calculateRegressionError(getModel(),p);
            fr.setTotalWeight(e);
            max = Math.max(max,e);
        }

        for(FeatureRank fr:getFeatures()) {
            fr.setImportancePercent(fr.getTotalWeight()/max);
        }
    }

    public GenerateRandom getRnd() {
        return rnd;
    }

    public void setRnd(GenerateRandom rnd) {
        this.rnd = rnd;
    }
}
