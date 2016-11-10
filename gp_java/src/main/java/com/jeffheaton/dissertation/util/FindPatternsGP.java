package com.jeffheaton.dissertation.util;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.prg.EncogProgram;
import org.encog.ml.prg.EncogProgramContext;
import org.encog.ml.prg.ProgramNode;
import org.encog.ml.prg.extension.FunctionFactory;
import org.encog.ml.prg.extension.StandardExtensions;
import org.encog.ml.prg.train.PrgPopulation;
import org.encog.ml.tree.TreeNode;
import org.encog.parse.expression.ExpressionNodeType;
import org.encog.util.Format;

import java.util.*;

public class FindPatternsGP {

    static public class FoundPattern implements Comparable<FoundPattern> {
        private String pattern;
        private int count;

        public FoundPattern(String thePattern) {
            this.pattern = thePattern;
            this.count = 1;
        }

        public String getPattern() {
            return this.pattern;
        }

        public int getCount() {
            return this.count;
        }

        public void increase() {
            this.count++;
        }

        @Override
        public int compareTo(FoundPattern o) {
            return Integer.compare(o.getCount(),this.count);
        }

        public boolean equals(FoundPattern o) {
            return this.compareTo(o)==0;
        }

        @Override
        public String toString() {
            StringBuilder result = new StringBuilder();
            result.append(this.count);
            result.append(",");
            result.append(this.pattern);
            return result.toString();
        }
    }

    private EncogProgram holder;
    private Set<String> varsFound = new HashSet<>();
    private HashMap<String,FoundPattern> patterns = new HashMap<>();
    private List<FoundPattern> patternSet = new ArrayList<>();

    public void find(final EncogProgram theHolder) {
        this.holder = theHolder;
        ProgramNode node = theHolder.getRootNode();
        findAtNode(node);
    }

    private void findAtNode(ProgramNode parent) {

        this.varsFound.clear();
        String foundExpression = renderAtNode(parent);

        if( this.varsFound.size()>=2 ) {
            if(this.patterns.containsKey(foundExpression)) {
                this.patterns.get(foundExpression).increase();
            } else {
                FoundPattern t = new FoundPattern(foundExpression);
                this.patterns.put(foundExpression,t);
                this.patternSet.add(t);
            }
        }

        for(TreeNode tn: parent.getChildNodes()) {
            ProgramNode child = (ProgramNode)tn;
            findAtNode(child);
        }
    }

    private String renderVar(ProgramNode node) {
        int varIndex = (int)node.getData()[0].toIntValue();
        String varName = this.holder.getVariables().getVariableName(varIndex);
        this.varsFound.add(varName);
        return varName;
    }

    private String renderFunction(ProgramNode node) {
        StringBuilder result = new StringBuilder();
        result.append(node.getName());
        result.append('(');
        for(int i=0;i<node.getChildNodes().size();i++) {
            if( i>0 ) {
                result.append(',');
            }
            ProgramNode childNode = node.getChildNode(i);
            result.append(renderAtNode(childNode));
        }
        result.append(')');
        return result.toString();
    }

    private String renderOperator(ProgramNode node) {
        StringBuilder result = new StringBuilder();

        String[] params = { renderAtNode(node.getChildNode(0)),
            renderAtNode(node.getChildNode(1)) };

        if( params[0].length()==0 && params[1].length()==0 ) {
            return "";
        }

        if( params[0].length()!=0 && params[1].length()==0) {
            return params[0];
        }

        if( params[0].length()==0 && params[1].length()!=0) {
            return params[1];
        }

        // Commute
        if( node.getName().equals("+") || node.getName().equals("*")) {
            Arrays.sort(params);
        }

        result.append("(");
        result.append(params[0]);
        result.append(node.getName());
        result.append(params[1]);
        result.append(")");
        return result.toString();
    }

    public ExpressionNodeType determineNodeType(ProgramNode node) {

        if (node.getName().equals("#const")) {
            return ExpressionNodeType.ConstVal;
        }

        if (node.getName().equals("#var")) {
            return ExpressionNodeType.Variable;
        }

        if( node.getChildNodes().size()!=2 ) {
            return ExpressionNodeType.Function;
        }

        String name = node.getName();

        if( !Character.isLetterOrDigit(name.charAt(0)) ) {
            return ExpressionNodeType.Operator;
        }

        return ExpressionNodeType.Function;
    }

    private String renderAtNode(ProgramNode node) {
        StringBuilder result = new StringBuilder();

        switch (determineNodeType(node)) {
            case Operator:
                result.append(renderOperator(node));
                break;
            case Variable:
                result.append(renderVar(node));
                break;
            case Function:
                result.append(renderFunction(node));
                break;
        }

        return result.toString();
    }

    public List<FoundPattern> getPatterns() {
        Collections.sort(this.patternSet);
        return this.patternSet;
    }

    public String reportString(int top, int total) {
        StringBuilder result = new StringBuilder();
        for(int i=0;i<Math.min(top,this.patternSet.size());i++) {
            if( i>0 ) {
                result.append(",");
            }
            FoundPattern p = this.patternSet.get(i);
            String pct = Format.formatDouble( ((double)p.getCount())/total,4);
            result.append(""+pct+":"+p.getPattern());
        }
        return result.toString();
    }

    public static void eval(String exp) {
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
        //f.compileExpression("((y/1)+x*2)/1");

        System.out.println("Expression: " + exp);
        f.compileExpression(exp);

        MLData x = new BasicMLData(3);
        x.setData(0,10);
        x.setData(1,11);
        x.setData(2,12);
        //System.out.println(f.compute(x));

        //ProgramNode root = f.getRootNode();
        //System.out.println(f.dumpAsCommonExpression());


        FindPatternsGP util = new FindPatternsGP();
        util.find(f);

        for(FoundPattern fp: util.getPatterns()) {
            System.out.println(fp);
        }
        System.out.println();
    }

    public static void main(String[] args) {
        eval("(x*y)");
        eval("(x*y)*z");
        eval("((y/1)+(x*2))/1");
    }
}
