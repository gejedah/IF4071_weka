/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package weka.filter;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;

public class WekaFilter {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception{
        FilteredClassifier tes = new FilteredClassifier();
        try{
            tes.setClassifier(new NaiveBayes());
            System.out.println("berhasil set classifier");
        } catch (Exception e) {
            System.out.println("set classifier gagal");
        }
        
        FilterDataset filter = new FilterDataset();
        filter.loadDataset("src/data/weather.nominal.arff");
        filter.setClassifier(tes);

//        evaluate() harus dijalankan sebelum build classifier
        filter.evaluate();
        filter.buildClassifier();
        filter.saveModel("src/data/training_data.model");

        //buat klasifikasi suatu data

        Klasifikasi cls = new Klasifikasi();

        cls.loadDataTest("src/data/test.csv");
        cls.loadModel("src/data/training_data.model");
        cls.classify();

    }
}
