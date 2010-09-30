package shared.test;

import java.io.File;
import java.io.PrintStream;

import shared.DataSet;
import shared.DataSetReader;
import shared.Instance;
import shared.filt.IndependentComponentAnalysis;
import util.linalg.Matrix;
import util.linalg.RectangularMatrix;

/**
 * A class for testing
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class IndepenentComponentAnalysisTest {
    
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
    	
    	
        try {
        	System.setOut(new PrintStream(new File("./icaoutput.txt")));
			Instance[] instances =  new Instance[100];
			for (int i = 0; i < instances.length; i++) 
			{
			    double[] data = new double[2];
			    data[0] = Math.sin(i/2.0);
			    data[1] = (Math.random() - .5)*2;
			    instances[i] = new Instance(data);
			}
			DataSetReader reader  = new DataSetReader("./data.csv");
			DataSet set = reader.read();
			
			//DataSet set = new DataSet(instances);
			System.out.println("Before randomizing");
			//System.out.println(set);
			Matrix projection = new RectangularMatrix(new double[][]{ {.6, .6}, {.4, .6}});
			for (int i = 0; i < set.size(); i++) 
			{
			    Instance instance = set.get(i);
			    instance.setData(projection.times(instance.getData()));
			}
			System.out.println("Before ICA");
			//System.out.println(set);
			IndependentComponentAnalysis filter = new IndependentComponentAnalysis(set, 10);
			filter.filter(set);
			System.out.println("After ICA");
			System.out.println(set);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
          
    }

}