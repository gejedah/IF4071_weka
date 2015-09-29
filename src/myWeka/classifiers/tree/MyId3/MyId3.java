package weka.classifiers.trees;

import weka.classifiers.Classifier;
import weka.core.*;

import java.util.Arrays;
import java.util.Enumeration;

/**
 * Created by hp on 9/28/2015.
 */
public class MyId3 extends Classifier {
    private Attribute mAttribute;
    private double mClassValue;
    private double[] mDistribution;
    private Attribute mClassAttribute;
    private MyId3[] mChild;
    private double mThreshold;
    private boolean isNumeric;

    public String globalInfo() {
        return  "Impelementasi dari algoritma ID3 untuk tugas IF4071."
                + "Dapat menghandle atribut nominal dan numerik. "
                + "Atribut kosong tidak diperbolehkan. ";
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES); // spek dari tugas
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }

    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        data = new Instances(data);
        data.deleteWithMissingClass();
        makeTree(data);
    }

    private void makeTree(Instances data) throws Exception {
        // jika tidak terdapat instance pada node ini
        if (data.numInstances() == 0) {
            mAttribute = null;
            mClassValue = Instance.missingValue();
            return;
        }

        double[] infoGains = new double[data.numAttributes()];
        double[] threshold = new double[data.numAttributes()];
        double maxIG = 0;
        double maxTR = 0;
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            if(att.isNumeric()) {
                Enumeration instEnum = data.enumerateInstances();
                double[] numericValue = new double[data.numInstances()];
                int i=0;
                while (instEnum.hasMoreElements()) {
                    Instance inst = (Instance) instEnum.nextElement();
                    numericValue[i] = inst.value(att.index());
                    i++;
                }
                Arrays.sort(numericValue);

                double maxInformationGain = 0;
                double thresholdMaxInformationGain = 0;
                for(int k=1; k<data.numInstances(); k++){
                    double temp = computeInfoGain(data, att, (numericValue[i] + numericValue[i-1])/2);
                    if (temp > maxInformationGain){
                        maxInformationGain = temp;
                        thresholdMaxInformationGain = (numericValue[i] + numericValue[i-1])/2;
                    }
                }
                infoGains[att.index()] = maxInformationGain;
                threshold[att.index()] = thresholdMaxInformationGain;

                if (infoGains[att.index()] > maxIG) {
                    maxIG = infoGains[att.index()];
                    maxTR = threshold[att.index()];
                    mAttribute = att;
                    isNumeric = true;
                    mThreshold = maxTR;
                }
            } else {
                infoGains[att.index()] = computeInfoGain(data, att);
                if (infoGains[att.index()] > maxIG) {
                    maxIG = infoGains[att.index()];
                    mAttribute = att;
                    isNumeric = false;
                }
            }
        }

        //mAttribute = data.attribute(Utils.maxIndex(infoGains));

        if (Utils.eq(maxIG, 0)) {
            mAttribute = null;
            mDistribution = new double[data.numClasses()];
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                mDistribution[(int) inst.classValue()]++;
            }
            Utils.normalize(mDistribution);
            mClassValue = Utils.maxIndex(mDistribution);
            mClassAttribute = data.classAttribute();
        } else {
            if(mAttribute.isNumeric()) {
                Instances[] splitData = splitData(data, mAttribute, mThreshold);
                mChild = new MyId3[2];
                for (int j = 0; j < 2; j++) {
                    mChild[j] = new MyId3();
                    mChild[j].makeTree(splitData[j]);
                }
            } else {
                Instances[] splitData = splitData(data, mAttribute);
                mChild = new MyId3[mAttribute.numValues()];
                for (int j = 0; j < mAttribute.numValues(); j++) {
                    mChild[j] = new MyId3();
                    mChild[j].makeTree(splitData[j]);
                }
            }
        }
    }

    public double classifyInstance(Instance instance)
            throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Missing Value tidak diperbolehkan.");
        }
        if (mAttribute == null) {
            return mClassValue;
        } else {
            if(isNumeric) {
                if(instance.value(mAttribute) < mThreshold) {
                    return mChild[0].classifyInstance(instance);
                } else {
                    return mChild[1].classifyInstance(instance);
                }

            } else {
                return mChild[(int) instance.value(mAttribute)].classifyInstance(instance);
            }
        }
    }

    public double[] distributionForInstance(Instance instance)
            throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("Missing Value tidak diperbolehkan.");
        }
        if (mAttribute == null) {
            return mDistribution;
        } else {
            return mChild[(int) instance.value(mAttribute)].
                    distributionForInstance(instance);
        }
    }

    private double computeInfoGain(Instances data, Attribute att)
            throws Exception {
        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double) splitData[j].numInstances() /
                        (double) data.numInstances()) *
                        computeEntropy(splitData[j]);
            }
        }
        return infoGain;
    }

    private double computeInfoGain(Instances data, Attribute att, double threshold)
            throws Exception {
        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att, threshold);
        for (int j = 0; j < 2; j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double) splitData[j].numInstances() /
                        (double) data.numInstances()) *
                        computeEntropy(splitData[j]);
            }
        }
        return infoGain;
    }

    private double computeEntropy(Instances data) throws Exception {
        double [] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        double entropy = 0;
        int j = 0;
        while (j < data.numClasses()){
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]/data.numInstances());
            }
            j++;
        }
        entropy /= (double) data.numInstances();
        return entropy;
    }

    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitData[(int) inst.value(att)].add(inst);
        }
        for (int i = 0; i < splitData.length; i++) {
            splitData[i].compactify();
        }
        return splitData;
    }

    private Instances[] splitData(Instances data, Attribute att, double threshold) {
        Instances[] splitData = new Instances[2];
        splitData[0] = new Instances(data, data.numInstances());
        splitData[1] = new Instances(data, data.numInstances());

        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            if (inst.value(att) < threshold) {
                splitData[0].add(inst);
            } else {
                splitData[1].add(inst);
            }
        }
        for (int i = 0; i < splitData.length; i++) {
            splitData[i].compactify();
        }
        return splitData;
    }

    public static void main(String[] args) {
        runClassifier(new MyId3(), args);
    }
}
