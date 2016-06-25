package com.jeffheaton.dissertation.features;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.prg.EncogProgram;
import org.encog.ml.prg.EncogProgramContext;
import org.encog.ml.prg.ProgramNode;
import org.encog.ml.prg.extension.FunctionFactory;
import org.encog.ml.prg.extension.StandardExtensions;
import org.encog.ml.prg.train.PrgPopulation;

/**
 * Created by jeffh on 6/25/2016.
 */
public class FindPatternsGP {

    public static void main(String[] args) {
        EncogProgramContext context = new EncogProgramContext();
        context.defineVariable("x");
        context.defineVariable("y");
        context.defineVariable("z");

        FunctionFactory factory = context.getFunctions();
        factory.addExtension(StandardExtensions.EXTENSION_VAR_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_CONST_SUPPORT);
        factory.addExtension(StandardExtensions.EXTENSION_NEG);
        factory.addExtension(StandardExtensions.EXTENSION_ADD);
        factory.addExtension(StandardExtensions.EXTENSION_SUB);
        factory.addExtension(StandardExtensions.EXTENSION_MUL);
        factory.addExtension(StandardExtensions.EXTENSION_DIV);
        factory.addExtension(StandardExtensions.EXTENSION_POWER);

        PrgPopulation pop = new PrgPopulation(context, 100);
        EncogProgram f = (EncogProgram)pop.getGenomeFactory().factor();
        f.setPopulation(pop);
        f.compileExpression("(x+y)/z");

        MLData x = new BasicMLData(3);
        x.setData(0,10);
        x.setData(1,11);
        x.setData(2,12);
        System.out.println(f.compute(x));

        ProgramNode root = f.getRootNode();
        System.out.println(root);
    }
}
