package myWeka.classifiers.tree;


import weka.classifiers.trees.j48.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Utils;

/**
 * Created by Fauzan Hilmi on 29/09/2015.
 */
public class myC45PruneableClassifierTree extends ClassifierTree {


    boolean m_pruneTheTree = false;
    float m_CF = 0.25f;
    boolean m_subtreeRaising = true;
    boolean m_cleanup = true;
    protected ModelSelection m_toSelectModel;
    protected ClassifierSplitModel m_localModel;
    protected ClassifierTree [] m_sons;
    protected boolean m_isLeaf;
    protected boolean m_isEmpty;
    protected Instances m_train;
    protected Distribution m_test;

    public myC45PruneableClassifierTree(ModelSelection toSelectLocModel,
                                      boolean pruneTree,float cf,
                                      boolean raiseTree,
                                      boolean cleanup)
            throws Exception {

        super(toSelectLocModel);

        m_toSelectModel = toSelectLocModel;
        m_pruneTheTree = pruneTree;
        m_CF = cf;
        m_subtreeRaising = raiseTree;
        m_cleanup = cleanup;
    }

    public myC45PruneableClassifierTree(ModelSelection ms) {
        super(ms);
        m_toSelectModel = ms;
    }

    public void buildClassifier(Instances data) throws Exception {

        // can classifier tree handle the data?
        getCapabilities().testWithFail(data);

        // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        buildTree(data);
        collapse();
        if (m_pruneTheTree) {
            prune();
        }
        if (m_cleanup) {
            cleanup(new Instances(data, 0));
        }
    }

    public void buildTree(Instances data) throws Exception {
        Instances [] localInstances;

        m_test = null;
        m_isLeaf = false;
        m_isEmpty = false;
        m_sons = null;System.out.println("masuk");

        m_localModel = m_toSelectModel.selectModel(data);
        if(m_localModel.numSubsets() > 1) {
            localInstances = m_localModel.split(data);
            data = null;
            m_sons = new myC45PruneableClassifierTree [m_localModel.numSubsets()];
            for(int i=0; i<m_sons.length; i++) {
                m_sons[i] = getNewTree(localInstances[i]);
                localInstances[i] = null;
            }
        }
        else {
            m_isLeaf = true;
            if (Utils.eq(data.sumOfWeights(), 0)) {
                m_isEmpty = true;
            }
            data = null;
        }
    }

    protected myC45PruneableClassifierTree getNewTree(Instances data) throws Exception {
        myC45PruneableClassifierTree newTree = new myC45PruneableClassifierTree(m_toSelectModel);
        newTree.buildTree(data, false);

        return newTree;
    }

    public final void collapse(){

        double errorsOfSubtree;
        double errorsOfTree;
        int i;

        if (!m_isLeaf){
            errorsOfSubtree = getTrainingErrors();
            errorsOfTree = localModel().distribution().numIncorrect();
            if (errorsOfSubtree >= errorsOfTree-1E-3){

                // Free adjacent trees
                m_sons = null;
                m_isLeaf = true;

                // Get NoSplit Model for tree.
                //TES
//                m_localModel.copyClassifierSplitModel(new NoSplit(localModel().distribution()));
                m_localModel = new NoSplit(localModel().distribution());
            }else
                for (i=0;i<m_sons.length;i++)
                    son(i).collapse();
        }
    }

    public void prune() throws Exception {

        double errorsLargestBranch;
        double errorsLeaf;
        double errorsTree;
        int indexOfLargestBranch;
        myC45PruneableClassifierTree largestBranch;
        int i;

        if (!m_isLeaf){

            // Prune all subtrees.
            for (i=0;i<m_sons.length;i++)
                son(i).prune();

            // Compute error for largest branch
            indexOfLargestBranch = localModel().distribution().maxBag();
            if (m_subtreeRaising) {
                errorsLargestBranch = son(indexOfLargestBranch).
                        getEstimatedErrorsForBranch((Instances)m_train);
            } else {
                errorsLargestBranch = Double.MAX_VALUE;
            }

            // Compute error if this Tree would be leaf
            errorsLeaf =
                    getEstimatedErrorsForDistribution(localModel().distribution());

            // Compute error for the whole subtree
            errorsTree = getEstimatedErrors();

            // Decide if leaf is best choice.
            if (Utils.smOrEq(errorsLeaf,errorsTree+0.1) &&
                    Utils.smOrEq(errorsLeaf,errorsLargestBranch+0.1)){

                // Free son Trees
                m_sons = null;
                m_isLeaf = true;

                // Get NoSplit Model for node.
                //TES
//                m_localModel.copyClassifierSplitModel(new NoSplit(localModel().distribution()));
                m_localModel = new NoSplit(localModel().distribution());
                return;
            }

            // Decide if largest branch is better choice
            // than whole subtree.
            if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)){
                largestBranch = son(indexOfLargestBranch);
                m_sons = largestBranch.m_sons;
                m_localModel = largestBranch.localModel();
                m_isLeaf = largestBranch.m_isLeaf;
                newDistribution(m_train);
                prune();
            }
        }
    }

    private double getEstimatedErrors(){

        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(localModel().distribution());
        else{
            for (i=0;i<m_sons.length;i++)
                errors = errors+son(i).getEstimatedErrors();
            return errors;
        }
    }

    private double getEstimatedErrorsForBranch(Instances data)
            throws Exception {

        Instances [] localInstances;
        double errors = 0;
        int i;

        if (m_isLeaf)
            return getEstimatedErrorsForDistribution(new Distribution(data));
        else{
            Distribution savedDist = localModel().distribution();
            localModel().resetDistribution(data);
            localInstances = (Instances[])localModel().split(data);
            //TES
//            localModel().distribution() = savedDist;
            for (i=0;i<m_sons.length;i++)
                errors = errors+
                        son(i).getEstimatedErrorsForBranch(localInstances[i]);
            return errors;
        }
    }

    private double getEstimatedErrorsForDistribution(Distribution
                                                             theDistribution){

        if (Utils.eq(theDistribution.total(),0))
            return 0;
        else
            return theDistribution.numIncorrect()+
                    Stats.addErrs(theDistribution.total(),
                            theDistribution.numIncorrect(),m_CF);
    }

    private double getTrainingErrors(){

        double errors = 0;
        int i;

        if (m_isLeaf)
            return localModel().distribution().numIncorrect();
        else{
            for (i=0;i<m_sons.length;i++)
                errors = errors+son(i).getTrainingErrors();
            return errors;
        }
    }

    private void newDistribution(Instances data) throws Exception {

        Instances [] localInstances;

        localModel().resetDistribution(data);
        m_train = data;
        if (!m_isLeaf){
            localInstances =
                    (Instances [])localModel().split(data);
            for (int i = 0; i < m_sons.length; i++)
                son(i).newDistribution(localInstances[i]);
        } else {

            // Check whether there are some instances at the leaf now!
            if (!Utils.eq(data.sumOfWeights(), 0)) {
                m_isEmpty = false;
            }
        }
    }

    private ClassifierSplitModel localModel(){
        return (ClassifierSplitModel)m_localModel;
    }

    private myC45PruneableClassifierTree son(int index){

        return (myC45PruneableClassifierTree)m_sons[index];
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 8986 $");
    }

    public double classifyInstance(Instance instance)
            throws Exception {

        double maxProb = -1;
        double currentProb;
        int maxIndex = 0;
        int j;

        for (j = 0; j < instance.numClasses(); j++) {
            currentProb = getProbs(j, instance, 1);
            if (Utils.gr(currentProb,maxProb)) {
                maxIndex = j;
                maxProb = currentProb;
            }
        }

        return (double)maxIndex;
    }

    private double getProbs(int classIndex, Instance instance, double weight)
            throws Exception {

        double prob = 0;

        if (m_isLeaf) {
            return weight * localModel().classProb(classIndex, instance, -1);
        } else {

            int treeIndex = localModel().whichSubset(instance);
            if (treeIndex == -1) {
                double[] weights = localModel().weights(instance);
                for (int i = 0; i < m_sons.length; i++) {
                    if (!son(i).m_isEmpty) {
                        prob += son(i).getProbs(classIndex, instance,
                                weights[i] * weight);
                    }
                }
                return prob;
            } else {
                System.out.println(treeIndex);
                if (son(treeIndex).m_isEmpty) {
                    return weight * localModel().classProb(classIndex, instance,
                            treeIndex);
                } else {
                    return son(treeIndex).getProbs(classIndex, instance, weight);
                }
            }
        }
    }
}
