package myWeka.classifiers.tree;

import weka.classifiers.Classifier;
        import weka.classifiers.trees.j48.*;
        import weka.core.*;

/**
 * Created by Fauzan Hilmi on 29/09/2015.
 */
public class MyJ48 extends Classifier {

    private myC45PruneableClassifierTree m_root;
    private boolean m_unpruned = false;
    private float m_CF = 0.25f;
    private int m_minNumObj = 2;
    private boolean m_subtreeRaising = true;
    private boolean m_noCleanup = false;

    public void MyJ48() {
    }

    @Override
    public void buildClassifier(Instances instances)
            throws Exception {

        ModelSelection modSelection;

        modSelection = new C45ModelSelection(m_minNumObj, instances);
        m_root = new myC45PruneableClassifierTree(modSelection, !m_unpruned, m_CF, m_subtreeRaising, !m_noCleanup);
        m_root.buildClassifier(instances);
    }

    public double classifyInstance(Instance instance) throws Exception {
        return m_root.classifyInstance(instance);
    }


}