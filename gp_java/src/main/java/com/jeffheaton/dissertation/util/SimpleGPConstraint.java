package com.jeffheaton.dissertation.util;

import org.encog.ml.ea.genome.Genome;
import org.encog.ml.ea.rules.ConstraintRule;
import org.encog.ml.prg.EncogProgram;
import org.encog.ml.prg.ProgramNode;
import org.encog.ml.tree.TreeNode;

import java.util.HashMap;
import java.util.Map;

public class SimpleGPConstraint implements ConstraintRule {
    /**
     * Is this genome valid?
     *
     * @param genome The genome.
     * @return True, if valid.
     */
    @Override
    public boolean isValid(Genome genome) {
        EncogProgram prg = (EncogProgram)genome;
        Map<String,Integer> counts = new HashMap<>();
        helper(counts, prg.getRootNode());
        if( counts.size()<1 ) {
            return false;
        }

        return true;
    }

    private void helper(Map<String,Integer> counts, ProgramNode parentNode) {
        if( parentNode.isVariable() ) {
            int varIndex = (int)parentNode.getData()[0].toIntValue();
            String name = parentNode.getOwner().getVariables().getVariableName(varIndex);
            if( counts.containsKey(name)) {
                counts.put(name, counts.get(name)+1);
            } else {
                counts.put(name,1);
            }
        }

        for(TreeNode childNode: parentNode.getChildNodes()) {
            helper(counts, (ProgramNode) childNode);
        }
    }
}
