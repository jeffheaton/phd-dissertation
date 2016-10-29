package com.jeffheaton.dissertation.autofeatures;

import com.jeffheaton.dissertation.util.SimpleGPConstraint;
import org.encog.mathutil.randomize.generate.GenerateRandom;
import org.encog.mathutil.randomize.generate.MersenneTwisterGenerateRandom;
import org.encog.ml.MLMethod;
import org.encog.ml.TrainingImplementationType;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.opp.selection.TournamentSelection;
import org.encog.ml.ea.population.Population;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.ml.prg.EncogProgram;
import org.encog.ml.prg.EncogProgramContext;
import org.encog.ml.prg.ProgramNode;
import org.encog.ml.prg.expvalue.ExpressionValue;
import org.encog.ml.prg.extension.FunctionFactory;
import org.encog.ml.prg.extension.StandardExtensions;
import org.encog.ml.prg.generator.RampedHalfAndHalf;
import org.encog.ml.prg.train.PrgPopulation;
import org.encog.ml.train.BasicTraining;
import org.encog.neural.networks.training.propagation.TrainingContinuation;

import java.util.List;
import java.util.Random;

/**
 * Created by jeffh on 10/29/2016.
 */
public class AutoFeatureGP extends BasicTraining {

    private final String[] names;
    private int populationSize = 100;
    private EncogProgramContext context;
    private PrgPopulation population;
    private final DumpFeatures dumpFeatures;
    private boolean includeOrigionalFeatures = false;
    private GenerateRandom rnd = new MersenneTwisterGenerateRandom();
    private FunctionFactory factory;

    public AutoFeatureGP(MLDataSet theDataset) {
        super(TrainingImplementationType.Iterative);
        setTraining(theDataset);
        this.names = new String[getTraining().getInputSize()];
        this.dumpFeatures = new DumpFeatures(theDataset);
    }

    private void initNames() {
        for(int i=1;i<=this.names.length;i++) {
            this.names[i-1] = "x"+i;
        }
    }

    private int antiSelect() {
        for(int t=0;t<5;t++) {
            int i = rnd.nextInt(this.populationSize);
        }
        return -1;
    }

    public void init() {
        if( this.names[0]==null) {
            initNames();
        }

        this.context = new EncogProgramContext();

        for(String str: this.names) {
            this.context.defineVariable(str);
        }

        //StandardExtensions.createNumericOperators(context);

        this.factory = this.context.getFunctions();
        factory.addExtension(StandardExtensions.EXTENSION_VAR_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_CONST_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_NEG);
        factory.addExtension(StandardExtensions.EXTENSION_ADD);
        factory.addExtension(StandardExtensions.EXTENSION_SUB);
        factory.addExtension(StandardExtensions.EXTENSION_MUL);
        //factory.addExtension(StandardExtensions.EXTENSION_POWER);
        factory.addExtension(StandardExtensions.EXTENSION_PDIV);

        this.population = new PrgPopulation(context,this.populationSize);
        this.population.getRules().addConstraintRule(new SimpleGPConstraint());
        (new RampedHalfAndHalf(context,1, 6)).generate(new Random(), this.population);

        if( this.includeOrigionalFeatures ) {
            forceOrigInputFeatures();
        }

        EvaluateFeatures score = new EvaluateFeatures(getTraining(), this);

        for(int i=0;i<1;i++) {
            System.out.println(score.calculateScores());
            this.dumpFeatures.dumpFeatures(i, this.population);
        }

    }

    private void forceOrigInputFeatures() {
        boolean[] vars = new boolean[getTraining().getInputSize()];
        List<Genome> list = this.population.flatten();

        for (Genome genome : list) {
            EncogProgram prg = ((EncogProgram) genome);
            ProgramNode root = prg.getRootNode();

            if (root.getTemplate() == StandardExtensions.EXTENSION_VAR_SUPPORT) {
                int varIndex = (int)root.getData()[0].toIntValue();
                vars[varIndex] = true;
            }
        }

        for(int i=0;i<vars.length;i++) {
            if( !vars[i] ) {
                final EncogProgram program = new EncogProgram(this.context);
                ProgramNode root = factory.factorProgramNode(StandardExtensions.EXTENSION_VAR_SUPPORT, program, new ProgramNode[]{});
                root.getData()[0] = new ExpressionValue(i);
                program.setRootNode(root);
                this.population.getSpecies().get(0).add(program);
            }
        }
    }

    public String[] getNames() {
        return names;
    }

    public int getPopulationSize() {
        return populationSize;
    }

    public void setPopulationSize(int populationSize) {
        this.populationSize = populationSize;
    }

    @Override
    public boolean canContinue() {
        return false;
    }

    @Override
    public TrainingContinuation pause() {
        return null;
    }

    @Override
    public void resume(TrainingContinuation state) {

    }

    @Override
    public MLMethod getMethod() {
        return null;
    }

    public DumpFeatures getDumpFeatures() {
        return this.dumpFeatures;
    }

    public boolean getIncludeOrigionalFeatures() {
        return this.includeOrigionalFeatures;
    }


    @Override
    public void iteration() {
        if( this.context == null) {
            init();
        }

        preIteration();

        postIteration();
    }

    public PrgPopulation getPopulation() {
        return population;
    }


}
