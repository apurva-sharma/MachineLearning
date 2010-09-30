/**
 * 
 */
package pparekh3;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.FixedIterationTrainer;
import shared.Instance;
import shared.SumOfSquaresError;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import func.nn.backprop.StochasticBackPropagationTrainer;

/**
 * @author Parth
 *
 */
public class NN {

	/**
	 * @param args
	 */
	public static void main(String[] args)
	{
		LoadDataSet trainingLoader = new LoadDataSet(LoadDataSet.TRAINING_SET_PATH, 1036);
		double[][][]trainingData = trainingLoader.getDataSet();
		
		LoadDataSet testLoader = new LoadDataSet(LoadDataSet.TEST_SET_PATH, 408);
		double[][][]testData = testLoader.getDataSet();
		
		BackPropagationNetworkFactory factory = 
            new BackPropagationNetworkFactory();
		
		Instance[] trainingPatterns = new Instance[trainingData.length];
        for (int i = 0; i < trainingPatterns.length; i++) {
        	trainingPatterns[i] = new Instance(trainingData[i][0]);
        	trainingPatterns[i].setLabel(new Instance(trainingData[i][1]));
        }
        
        
        Instance[] testPatterns = new Instance[testData.length];
        for (int i = 0; i < testPatterns.length; i++) {
        	testPatterns[i] = new Instance(testData[i][0]);
        	testPatterns[i].setLabel(new Instance(testData[i][1]));
        }
        
        BackPropagationNetwork network = factory.createClassificationNetwork(
           new int[] { 16, 21, 26 });
        ErrorMeasure measure = new SumOfSquaresError();
        DataSet set = new DataSet(trainingPatterns);
        
        //NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(
            //set, network, measure);
        //OptimizationAlgorithm o = new RandomizedHillClimbing(nno);
        //OptimizationAlgorithm o = new SimulatedAnnealing(1E12, .95, nno);
        
        //BatchBackPropagationTrainer trainer = new BatchBackPropagationTrainer(set, network, new SumOfSquaresError(), new RPROPUpdateRule());
        StochasticBackPropagationTrainer trainer = new StochasticBackPropagationTrainer(set, network, new SumOfSquaresError(), new RPROPUpdateRule());
        
        int increment = 10;
        
        for(int noOfEpochs = 500 ; noOfEpochs <= 5000 ; noOfEpochs += 500)
        {
        	
        	
        	long startTime = System.currentTimeMillis();
	        FixedIterationTrainer fit = new FixedIterationTrainer(trainer, noOfEpochs);
	        fit.train();
	        
	        //Instance opt = o.getOptimal();
	        //network.setWeights(opt.getData());
	        
		      //Training Error..
		        int trainingIncorrectClassification = 0;
		        for (int i = 0; i < trainingPatterns.length; i++)
		        {
		            network.setInputValues(trainingPatterns[i].getData());
		            network.run();
		            
		            int actualLabel = trainingPatterns[i].getLabel().getData().argMax();
		            int predictedLabel = network.getOutputValues().argMax(); 
		            
		            if(actualLabel != predictedLabel)
		            {
		            	trainingIncorrectClassification++;
		            }
		            
		            //System.out.println("~~" + i);
		            //System.out.println(patterns[i].getLabel().getData().argMax());
		            //System.out.println(network.getOutputValues().argMax());
		        }
		        double trainingError = (trainingIncorrectClassification * 100) / trainingPatterns.length;
		        
		        System.out.println(noOfEpochs + " " + trainingError);	        
		        
		        BufferedWriter out;
				try
				{
					out = new BufferedWriter(new FileWriter("NNSocTrain.result", true));
			        out.write(noOfEpochs + " " + trainingError + "\n");
			        out.close();
				}
				catch (IOException e)
				{
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	        
		      //Test Error..
		        int testIncorrectClassification = 0;
		        for (int i = 0; i < testPatterns.length; i++)
		        {
		            network.setInputValues(testPatterns[i].getData());
		            network.run();
		            
		            int actualLabel = testPatterns[i].getLabel().getData().argMax();
		            int predictedLabel = network.getOutputValues().argMax(); 
		            
		            if(actualLabel != predictedLabel)
		            {
		            	testIncorrectClassification++;
		            }
		            
		            //System.out.println("~~" + i);
		            //System.out.println(patterns[i].getLabel().getData().argMax());
		            //System.out.println(network.getOutputValues().argMax());
		        }
		        
		        double testError = (testIncorrectClassification * 100) / testPatterns.length;
	        
		        System.out.println(noOfEpochs + " " + testError);	        
		        
		        //BufferedWriter out;
				try
				{
					out = new BufferedWriter(new FileWriter("NNSocTest.result", true));
			        out.write(noOfEpochs + " " + testError + "\n");
			        out.close();
				}
				catch (IOException e)
				{
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			
	        
        }
        
	}

}
