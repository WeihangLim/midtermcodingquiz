package ai0209.Q2;

import ai.certifai.solution.modelsaveload.MnistImageSave;
import com.sun.xml.internal.bind.v2.TODO;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class Mnistq2 {

    private static int labelIndex = 784;
    private static int numClasses = 10;
    final static int seed = 1234;
    final static int batchSize = 500;
    final static int epoch = 5;

    private static Logger log = LoggerFactory.getLogger(MnistImageSave.class);

    public static void main(String[] args) throws IOException, Exception {

        int height = 70000;
        int width = 784;
        int channels = 256;

        File inputFile = new ClassPathResource("ai0209/Q2/mnist_784_csv.csv").getFile();

        FileSplit fileSplit = new FileSplit(inputFile);
        RecordReader rr = new CSVRecordReader(1, ',');
//        rr.initialize(new FileSplit(inputFile));

        DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
        MnistDataSetIterator trainMnist = new MnistDataSetIterator(batchSize,true, seed);
        MnistDataSetIterator testMnist = new MnistDataSetIterator(batchSize,false, seed);
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler(0,1);

        scaler.fit(trainMnist);
        trainMnist.setPreProcessor(scaler);
        testMnist.setPreProcessor(scaler);

        //Model Config
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
//                .dropOut(0.5)
                .activation(Activation.RELU)
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(trainMnist.inputColumns())
                        .nOut(128)
                        .build())
                .layer(1,new DenseLayer.Builder()
                        .nOut(256)
                        .build())
                .layer(2,new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(3,new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(4,new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nOut(trainMnist.totalOutcomes())
                        .build())
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        InMemoryStatsStorage storage = new InMemoryStatsStorage();

//        UIServer server = UIServer.getInstance();
//        server.attach(storage);
//        model.setListeners(new StatsListener(storage), new ScoreIterationListener(1000));
        log.info("*****TRAIN MODEL********");
        for (int i = 0; i <= epoch; i++) {
            model.fit(trainMnist);

        }

        log.info("******SAVE TRAINED MODEL******");

        File SaveFile = new File(System.getProperty("java.io.tmpdir"));
        log.info(SaveFile.toString());

        NativeImageLoader nativeImageLoader = new NativeImageLoader(height, width); //28x28
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1); //translate image into seq of 0..1 input values

        Evaluation evalTrain = model.evaluate(trainMnist);
        Evaluation evalTest = model.evaluate(testMnist);

        System.out.println(evalTrain.stats());
        System.out.println(evalTest.stats());
        System.in.read();

    }
}
