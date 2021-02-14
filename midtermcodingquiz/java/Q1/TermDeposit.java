package ai0209.Q1;

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

public class TermDeposit {

    final static int seed = 1234;
    final static int epoch = 5;
    final static int batchSize = 1000;

    public static void main(String[] args) throws IOException, InterruptedException {

        File trainFilePath = new ClassPathResource("ai0209/Q1/train.csv").getFile();

        CSVRecordReader trainCsvRR = new CSVRecordReader(1, ',');
        trainCsvRR.initialize(new FileSplit(trainFilePath));

        Schema trainSchema = getTrainSchema();


        TransformProcess trainTP;
        trainTP = new TransformProcess.Builder(trainSchema)
                .categoricalToInteger("job", "marital", "education", "default", "housing",
                        "loan", "contact", "month", "poutcome", "subscribed")
                .filter(new FilterInvalidValues())
                .build();

        List<List<Writable>> oriData = new ArrayList<>();

        while (trainCsvRR.hasNext()) {
            oriData.add(trainCsvRR.next());
        }

        List<List<Writable>> trainTransformedData = LocalTransformExecutor.execute(oriData, trainTP);

        System.out.println(trainTP.getFinalSchema());
        System.out.println(oriData.size());
        System.out.println(trainTransformedData.size());

        CollectionRecordReader cRR = new CollectionRecordReader(trainTransformedData);
        RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(cRR,
                trainTransformedData.size(), 17, 2);
        DataSet dataSet = dataIter.next();
        SplitTestAndTrain split = dataSet.splitTestAndTrain(0.8);
        org.nd4j.linalg.dataset.DataSet trainSplit = split.getTrain();
        DataSet valSplit = split.getTest();
        trainSplit.setLabelNames(Arrays.asList("0", "1"));
        valSplit.setLabelNames(Arrays.asList("0", "1"));

        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainSplit);
        scaler.transform(trainSplit);
        scaler.transform(valSplit);

        ViewIterator trainIter = new ViewIterator(trainSplit, batchSize);
        ViewIterator testIter = new ViewIterator((org.nd4j.linalg.dataset.DataSet) valSplit, batchSize);

        HashMap<Integer, Double> scheduler = new HashMap<>();
        scheduler.put(0, 1e-3);
        scheduler.put(2, 1e-4);
        scheduler.put(3, 1e-5);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, scheduler)))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(trainIter.inputColumns())
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
                        .nOut(trainIter.totalOutcomes())
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(100));

        ArrayList<Double> trainloss = new ArrayList<>();
        ArrayList<Double> valloss = new ArrayList<>();
        DataSetLossCalculator trainLossCalc = new DataSetLossCalculator(trainIter, true);
        DataSetLossCalculator testLossCalc = new DataSetLossCalculator(testIter, true);

        for (int i = 0; i <= epoch; i++) {
            model.fit(trainIter);
            trainloss.add(trainLossCalc.calculateScore(model));
            valloss.add(testLossCalc.calculateScore(model));

        }

        Evaluation trainEval = model.evaluate(trainIter);
        Evaluation testEval = model.evaluate(testIter);

        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());

        ModelSerializer.writeModel(model, "C:\\Users\\Admin\\Desktop\\model\\termdeposit.zip", true);
        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new MinMaxSerializerStrategy());
        normalizerSerializer.write(scaler, "C:\\Users\\Admin\\Desktop\\model\\normalizer.zip");

        Nd4j.getEnvironment().allowHelpers(false);
        List<List<Writable>> valcollection = RecordConverter.toRecords(valSplit);
        INDArray valArray = RecordConverter.toMatrix(DataType.FLOAT, valcollection);
        INDArray valFeatures = valArray.getColumns(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

        List<String> prediction = model.predict(valSplit);
        INDArray output = model.output(valFeatures);

        for (int i = 0; i < 10; i++) {
            System.out.println("Prediction: " + prediction.get(i) + "; output:" + output.getRow(i));
        }

    }

    static Schema getTrainSchema() {

        return new Schema.Builder()
                .addColumnsInteger("ID", "age")
                .addColumnCategorical("job",
                        Arrays.asList("admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                                "retired", "self-employed", "services", "student", "technician",
                                "unemployed", "unknown"))
                .addColumnCategorical("marital", Arrays.asList("married", "divorced", "single"))
                .addColumnCategorical("education", Arrays.asList("unknown", "secondary", "tertiary", "primary"))
                .addColumnCategorical("default", Arrays.asList("no", "yes"))
                .addColumnDouble("balance")
                .addColumnCategorical("housing", Arrays.asList("no", "yes"))
                .addColumnCategorical("loan", Arrays.asList("no", "yes"))
                .addColumnCategorical("contact", Arrays.asList("telephone", "cellular", "unknown"))
                .addColumnInteger("day")
                .addColumnCategorical("month", Arrays.asList("jan", "feb", "mar", "apr", "may", "jun",
                        "jul", "aug", "sep", "oct", "nov", "dec"))
                .addColumnsInteger("duration", "campaign", "pdays", "previous")
                .addColumnCategorical("poutcome", Arrays.asList("unknown", "success", "failure", "other"))
                .addColumnCategorical("subscribed", "no", "yes")
                .build();
    }

}
