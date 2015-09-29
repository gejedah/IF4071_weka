/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package myWeka.filter;

import myWeka.classifiers.tree.myID3;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.supervised.instance.Resample;

import java.util.Scanner;

public class WekaFilter {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception{
        FilteredClassifier tes = new FilteredClassifier();
//        MultiClassClassifier tes = new MultiClassClassifier();
        Scanner in = new Scanner(System.in);
        System.out.println("1 untuk NaiveBayes");
        System.out.println("2 untuk J48");
        System.out.println("3 untuk ID3");
        System.out.println("4 untuk myC45");
        System.out.println("5 untuk myID3");
        int mode = 0;
        try{
            while ((mode < 1) || (mode > 5)){
                System.out.print("Mode yang diinginkan adalah: ");
                mode = in.nextInt();
                if (mode == 1){
                    tes.setClassifier(new NaiveBayes());
                }
                else if (mode == 2){
                    tes.setClassifier(new J48());
                }
                else if (mode == 3){
                    tes.setClassifier(new Id3());
                }
                else if (mode == 5){
                    tes.setClassifier(new myID3());
                }
            }
            System.out.println("berhasil set classifier");
        } catch (Exception e) {
            System.out.println("set classifier gagal");
        }

        FilterDataset filter = new FilterDataset();
        filter.loadDataset("src/data/weather.nominal.arff");
        Instances testSet = filter.percentageSplit(100);
//        filter.Resample();
        filter.setClassifier(tes);

//        evaluate() harus dijalankan sebelum build classifier
        filter.evaluate();
        filter.buildClassifier();
        filter.saveModel("src/data/training_data.model");

        //buat klasifikasi suatu data

        Klasifikasi cls = new Klasifikasi();

        cls.loadDataTest("src/data/test.arff");
        cls.loadModel("src/data/training_data.model");
        cls.classify();

    }
}
