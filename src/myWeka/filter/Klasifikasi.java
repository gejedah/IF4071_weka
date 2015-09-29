/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package myWeka.filter;

import com.opencsv.CSVReader;
import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.FastVector;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

import java.util.List;
import java.util.ArrayList;
import java.io.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by kevin on 9/27/15.
 */

public class Klasifikasi {
    private Instances instances;
    private FilteredClassifier classifier;

    public void loadDataTest(String DATA_SOURCE) {
        try {
            if (DATA_SOURCE.endsWith(".arff")){
                classifier = new FilteredClassifier();
                BufferedReader reader = new BufferedReader(new FileReader(DATA_SOURCE));
                ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);
                instances = arff.getData();
                instances.setClassIndex(instances.numAttributes() - 1);
                System.out.println("Byk Attribut: " + instances.numAttributes());
                System.out.println("Byk Instances: " + instances.numInstances());
                System.out.println("===== Loaded data test Arff: " + DATA_SOURCE + " =====");
                reader.close();
            }
            else if (DATA_SOURCE.endsWith(".csv")){
                CSVLoader csvLoader = new CSVLoader();
                csvLoader.setSource(new File(DATA_SOURCE));
                instances = csvLoader.getDataSet();
                instances.setClassIndex(instances.numAttributes()-1);
                System.out.println("Byk Attribut: "+ instances.numAttributes());
                System.out.println("Byk Instances: "+ instances.numInstances());
//                System.out.println(instances);
                System.out.println("===== Loaded data test CSV: " + DATA_SOURCE + " =====");
            }
        }
        catch (IOException e) {
            System.out.println("Problem found when reading: " + DATA_SOURCE);
        }
    }
    
    public String loadModel(String DATA_SOURCE) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(DATA_SOURCE));
            Object tmp = in.readObject();
            classifier = (FilteredClassifier) tmp;
            in.close();
            System.out.println("===== Loaded model: " + DATA_SOURCE + " =====");
            return "yes sucses";
        }
        catch (Exception e) {           
            System.out.println("Problem found when reading: " + DATA_SOURCE);
            return "Failure "+e.getMessage();
        }
    }

    public String testModel(){

        return null;
    }

    public void classify() {
        try {
            double pred = classifier.classifyInstance(instances.instance(0));
            System.out.println("===== Classified instance =====");
            System.out.println("Class predicted: " + instances.classAttribute().value((int) pred));
        }
        catch (Exception e) {
            System.out.println("Problem found when classifying the text");
        }
    }

    public String getKategori(){
        double pred;
        try {
            pred = classifier.classifyInstance(instances.instance(0));
            return instances.classAttribute().value((int) pred); 
        } catch (Exception ex) {
            Logger.getLogger(Klasifikasi.class.getName()).log(Level.SEVERE, null, ex);
        }
        return "Kategori Error"; 
    }
}
