package ai0209.Q1;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MinMaxSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class TermDep {

    final static int seed = 1234;
    final static int epoch = 5;
    final static int batchSize = 500;

    public static void main(String[] args) throws IOException, InterruptedException {

        File inputFile = new ClassPathResource("ai0209/Q1/train.csv").getFile();
        RecordReader rr = new CSVRecordReader(1,',');
        rr.initialize(new FileSplit(inputFile));

        Schema inputDataSchema = new Schema.Builder()
                .addColumnInteger("ID")
                .addColumnInteger("age")
                .addColumnCategorical("job",
                        Arrays.asList("admin.","unknown","services","management","technician","retired",
                        "blue-collar","housemaid","self-employed","student","entrepreneur","unemployed"))
                .addColumnCategorical("marital",Arrays.asList("married","divorced","single"))
                .addColumnCategorical("education",Arrays.asList("unknown","primary","secondary","tertiary"))
                .addColumnCategorical("default", Arrays.asList("no","yes"))
                .addColumnInteger("balance")
                .addColumnCategorical("housing",Arrays.asList("no","yes"))
                .addColumnCategorical("loan",Arrays.asList("no","yes"))
                .addColumnCategorical("contact", Arrays.asList("telephone","cellular","unknown"))
                .addColumnInteger("day")
                .addColumnCategorical("month",Arrays.asList("jan","feb","mar","apr","may","jun",
                        "jul","aug","sep","oct","nov","dec"))
                .addColumnInteger("duration")
                .addColumnInteger("campaign")
                .addColumnInteger("pdays")
                .addColumnInteger("previous")
                .addColumnCategorical("poutcome", Arrays.asList("unknown","success","failure","other"))
                .addColumnCategorical("subscribed", Arrays.asList("no","yes"))
                .build();

        System.out.println("Input data schema details:");
        System.out.println(inputDataSchema);

        System.out.println("\n\nOther information obtainable from schema:");
        System.out.println("Number of columns: " + inputDataSchema.numColumns());
        System.out.println("Column names: " + inputDataSchema.getColumnNames());
        System.out.println("Column types: " + inputDataSchema.getColumnTypes());


        TransformProcess traintp = new TransformProcess.Builder(inputDataSchema)
//                .removeColumns("ID")
//                .transform(new ReplaceInvalidWithIntegerTransform("age", 0))
                .categoricalToInteger("job","marital","education","default","housing","loan",
                        "contact","month","poutcome","subscribed")
                .filter(new FilterInvalidValues())
                .build();

        Schema outputSchema = traintp.getFinalSchema();
        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);


        List<List<Writable>> trainData = new ArrayList<>();

        while(rr.hasNext()){
            trainData.add(rr.next());
        }


        List<List<Writable>> processedData = LocalTransformExecutor.execute(trainData, traintp);
        System.out.println(trainData.size());
        System.out.println(processedData.size());

        CollectionRecordReader crr = new CollectionRecordReader(processedData);
        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(crr,processedData.size(),17,2);

        DataSet dataSet = iterator.next();

        SplitTestAndTrain stt = dataSet.splitTestAndTrain(0.75);
        org.nd4j.linalg.dataset.DataSet trainSplit = stt.getTrain();
        org.nd4j.linalg.dataset.DataSet testSplit = stt.getTest();

        trainSplit.setLabelNames(Arrays.asList("0","1"));
        testSplit.setLabelNames(Arrays.asList("0","1"));

        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainSplit);
        scaler.transform(trainSplit);
        scaler.transform(testSplit);

        ViewIterator trainIte = new ViewIterator(trainSplit, batchSize);
        ViewIterator testIte = new ViewIterator(testSplit, batchSize);

        HashMap<Integer, Double> scheduler = new HashMap<>();
        scheduler.put(0,1e-3);
        scheduler.put(2,1e-4);
        scheduler.put(3,1e-5);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, scheduler)))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(trainIte.inputColumns())
                        .nOut(256)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        .nOut(trainIte.totalOutcomes())
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();
        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(100));

        ArrayList<Double> trainLoss = new ArrayList<>();
        ArrayList<Double> testLoss = new ArrayList<>();
        DataSetLossCalculator trainLossCalc = new DataSetLossCalculator(trainIte, true);
        DataSetLossCalculator testlossCalc = new DataSetLossCalculator(testIte, true);

        for (int i = 0; i <= epoch; i++) {
            model.fit(trainIte);
            trainLoss.add(trainLossCalc.calculateScore(model));
            testLoss.add(testlossCalc.calculateScore(model));

        }

        Evaluation evalTrain = model.evaluate(trainIte);
        Evaluation evalTest = model.evaluate(testIte);

        System.out.println(evalTrain.stats());
        System.out.println(evalTest.stats());


        ModelSerializer.writeModel(model, "C:\\Users\\Admin\\Desktop\\model\\termdeposit.zip", true);
        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new MinMaxSerializerStrategy());
        normalizerSerializer.write(scaler,"C:\\Users\\Admin\\Desktop\\model\\normalizer.zip");


        Nd4j.getEnvironment().allowHelpers(false);
        List<List<Writable>> valcollection = RecordConverter.toRecords(testSplit);
        INDArray testArray = RecordConverter.toMatrix(DataType.FLOAT, valcollection);
        INDArray testFeatures = testArray.getColumns(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);

        List <String> prediction = model.predict(testSplit);
        INDArray output = model.output(testFeatures);

        for (int i=0; i<10; i++){
            System.out.println("Prediction:" + prediction.get(i) + "; Output:" + output.getRow(i));
        }

    }


}
