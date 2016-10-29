package com.jeffheaton.dissertation.autofeatures;

import au.com.bytecode.opencsv.CSVWriter;
import org.encog.EncogError;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.population.Population;
import org.encog.ml.prg.EncogProgram;
import org.encog.ml.prg.ProgramNode;
import org.encog.ml.prg.extension.StandardExtensions;
import org.encog.ml.tree.TreeNode;
import org.encog.util.Format;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * Created by jeffh on 10/29/2016.
 */
public class DumpFeatures {
    private File logFeatureDir;
    private MLDataSet trainingSet;

    public DumpFeatures(MLDataSet theTrainingSet) {
        this.logFeatureDir = null;
        this.trainingSet = theTrainingSet;
    }


    private void calculateStats(EncogProgram prg, double[] stats) {
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.POSITIVE_INFINITY;

        double sum = 0;
        for(MLDataPair pair: this.trainingSet) {
            MLData output = prg.compute(pair.getInput());
            double d = output.getData(0);
            max = Math.max(d,max);
            min = Math.min(d,min);
            sum+=d;
        }
        double mean = sum / this.trainingSet.size();

        sum = 0;
        for(MLDataPair pair: this.trainingSet) {
            MLData output = prg.compute(pair.getInput());
            double d = output.getData(0);
            double diff = mean - d;
            sum+=diff*diff;
        }
        double sdev = Math.sqrt(sum);

        stats[0] = min;
        stats[1] = max;
        stats[2] = mean;
        stats[3] = sdev;

    }

    public void dumpFeatures(int step, Population pop) {

        if( this.logFeatureDir !=null ) {
            CSVWriter writer = null;
            try {
                String filename = "autofeatures-" + step + ".csv";
                writer = new CSVWriter(new FileWriter(new File(this.getLogFeatureDir(), filename)));

                writer.writeNext(new String[]{"index", "score", "birth", "species", "vars", "min", "max", "mean", "sdev", "feature"});

                List<Genome> list = pop.flatten();
                int idx = 0;
                double[] stats = new double[4];

                for (Genome genome : list) {
                    EncogProgram prg = ((EncogProgram) genome);
                    calculateStats(prg, stats);

                    String[] line = {
                            "" + idx,
                            Format.formatDouble(genome.getScore(), 4),
                            "" + genome.getBirthGeneration(),
                            "" + pop.getSpecies().indexOf(genome.getSpecies()),
                            findVariables(prg),
                            Format.formatDouble(stats[0], 4),
                            Format.formatDouble(stats[1], 4),
                            Format.formatDouble(stats[2], 4),
                            Format.formatDouble(stats[3], 4),
                            prg.dumpAsCommonExpression()
                    };
                    writer.writeNext(line);
                    idx++;
                }

            } catch (IOException ex) {
                throw new EncogError(ex);
            } finally {
                try {
                    writer.close();
                } catch (IOException e) {
                    throw new EncogError(e);
                }
            }
        }
    }

    private void traverse(EncogProgram prg, TreeNode parentNode, Set<String> variables) {
        if( parentNode instanceof ProgramNode) {
            ProgramNode prgNode = (ProgramNode) parentNode;
            if (prgNode.getTemplate() == StandardExtensions.EXTENSION_VAR_SUPPORT) {
                int varIndex = (int)prgNode.getData()[0].toIntValue();
                String varName = prg.getVariables().getVariableName(varIndex);
                variables.add(varName);
            }
        }

        for(TreeNode childNode : parentNode.getChildNodes()) {
            traverse(prg,childNode,variables);
        }
    }

    private String findVariables(EncogProgram prg) {
        Set<String> set = new TreeSet<>();
        traverse(prg,prg.getRootNode(),set);
        return set.toString();
    }



    public File getLogFeatureDir() {
        return logFeatureDir;
    }

    public void setLogFeatureDir(File logFeatureDir) {
        this.logFeatureDir = logFeatureDir;
    }

}
