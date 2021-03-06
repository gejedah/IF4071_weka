/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package myWeka.filter;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.CSVLoader;
import weka.filters.supervised.instance.Resample;

import java.io.*;
import java.util.Random;

/**
 * Created by kevin on 9/27/15.
 */

public class FilterDataset {
    private Instances dataTrain;
    private FilteredClassifier classifier;
//    private MultiClassClassifier classifier;
    private Resample sampler;

    public void loadDataset(String DATA_SOURCE) {
        try {
            if (DATA_SOURCE.endsWith(".arff")){
                BufferedReader reader = new BufferedReader(new FileReader(DATA_SOURCE));
                ArffReader arff = new ArffReader(reader);
                dataTrain = arff.getData();
                System.out.println("Byk Attribut: "+dataTrain.numAttributes());
                System.out.println("Byk Instances: "+dataTrain.numInstances());
                dataTrain.setClassIndex(dataTrain.numAttributes()-1);
                System.out.println("Byk kelas: " + dataTrain.numClasses());
//                System.out.println(dataTrain);
                System.out.println("===== Loaded dataset Arff: " + DATA_SOURCE + " =====");
                reader.close();
            }
            else if (DATA_SOURCE.endsWith(".csv")){
                CSVLoader csvLoader = new CSVLoader();
                csvLoader.setSource(new File(DATA_SOURCE));
                dataTrain = csvLoader.getDataSet();
                System.out.println("Byk Attribut: " + dataTrain.numAttributes());
                System.out.println("Byk Instances: " + dataTrain.numInstances());
                dataTrain.setClassIndex(dataTrain.numAttributes()-1);
                System.out.println("Byk kelas: " +dataTrain.numClasses());
//                System.out.println(dataTrain);
                System.out.println("===== Loaded dataset CSV: " + DATA_SOURCE + " =====");
            }
        }
        catch (IOException e) {
            System.out.println("Problem found when reading: " + DATA_SOURCE);
        }
    }

//    Based on Index
    public void removeAttribute(int index){
        dataTrain.deleteAttributeAt(index);
    }

    public void Resample() throws Exception {
//        System.out.println("Sebelum Resample "+ dataTrain);
        dataTrain = dataTrain.resample(new Random(2));
//        System.out.println("Sesudah Resample " + dataTrain);
    }

    public Instances percentageSplit(int percent){
        int train_size = Math.round(dataTrain.numInstances()*percent/100);
        int test_size= dataTrain.numInstances() - train_size;
//        System.out.println("Train size: "+ train_size );
//        System.out.println("Test size: "+ test_size );
        Instances dataTest = new Instances(dataTrain, train_size-1, test_size);
        dataTrain = new Instances(dataTrain, 0, train_size);
//        System.out.println(dataTest);
//        System.out.println(dataTrain);
        return  dataTest;
    }

    public void setClassifier(Classifier c){
        try{
            classifier.setClassifier(c);
        } catch (Exception e){
            System.out.println("Problem found when set classifier");
        }
    }

    /**
     * @param classifier the classifier to set
     */
    public void setClassifier(FilteredClassifier classifier) {
//    public void setClassifier(MultiClassClassifier classifier) {
        this.classifier = classifier;
    }

    public void evaluate() {
        try {
            dataTrain.setClassIndex(dataTrain.numAttributes()-1);
//            StringToWordVector filter;
//            filter = new StringToWordVector();
//            filter.setAttributeIndices("first");
//            setClassifier(new FilteredClassifier());
//            classifier.setFilter(filter);
            Evaluation eval = new Evaluation(dataTrain);
//            eval.crossValidateModel(classifier, dataTrain, 10, new Random(1));
            Object hasil = new Object();
            eval.evaluateModel(classifier,dataTrain,hasil);
            System.out.println("Hasilnya: "+hasil);
//            eval.evaluateModel(classifier, dataTrain);
            double[][] tes = eval.confusionMatrix();
            System.out.println("Summary: " + eval.toSummaryString());
            System.out.println("Class Detail: "+ eval.toClassDetailsString());
            System.out.println("=== Confusion Matrix ===");
            for (int i=0; i< tes.length; i++){
                for (int j = 0; j < tes[i].length ; j++) {
                    System.out.print(tes[i][j] + " ");
                }
                System.out.println();
            }
            System.out.println();
            System.out.println("===== Evaluating on filtered (training) dataset done =====");
        }
        catch (Exception e) {
            System.out.println("Problem found when evaluating");
            //e.printStackTrace();
        }
    }

    public void buildClassifier() {
        try {
            dataTrain.setClassIndex(dataTrain.numAttributes()-1);
//            StringToWordVector filter;
//            filter = new StringToWordVector();
//            filter.setAttributeIndices("first");
//            setClassifier(new FilteredClassifier());
//            classifier.setFilter(filter);
            classifier.buildClassifier(dataTrain);
            // Uncomment to see the classifier
             System.out.println(classifier);
            System.out.println("===== classifier builded =====");
        }
        catch (Exception e) {
            System.out.println("Problem found when building");
        }
    }

    public void saveModel(String DATA_SOURCE) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(DATA_SOURCE));
            out.writeObject(classifier);
            out.close();
            System.out.println("===== Saved model: " + DATA_SOURCE + " =====");
        }
        catch (IOException e) {
            System.out.println("Problem found when writing: " + DATA_SOURCE);
        }
    }

    
}
