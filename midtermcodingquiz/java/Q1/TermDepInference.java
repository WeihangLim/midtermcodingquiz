//package ai0209.Q1;
//
//import org.datavec.api.records.reader.RecordReader;
//import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
//import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
//import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
//import org.datavec.api.split.FileSplit;
//import org.datavec.api.transform.TransformProcess;
//import org.datavec.api.transform.filter.FilterInvalidValues;
//import org.datavec.api.transform.schema.Schema;
//import org.datavec.api.transform.transform.integer.ReplaceInvalidWithIntegerTransform;
//import org.datavec.api.util.ndarray.RecordConverter;
//import org.datavec.api.writable.Writable;
//import org.datavec.local.transforms.LocalTransformExecutor;
//import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.deeplearning4j.util.ModelSerializer;
//import org.nd4j.common.io.ClassPathResource;
//import org.nd4j.linalg.api.buffer.DataType;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
//import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
//import org.nd4j.linalg.dataset.api.preprocessor.serializer.MinMaxSerializerStrategy;
//import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
//import org.nd4j.linalg.factory.Nd4j;
//
//import java.io.File;
//import java.io.FileWriter;
//import java.io.IOException;
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//
//class TermDepositInference {
//
//    public static void main(String[] args) throws Exception {
//
//        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("C:\\Users\\Admin\\Desktop\\model\\termdeposit.zip", true);
//
//        File inputFile = new ClassPathResource("ai0209/Q1/train.csv").getFile();
//        RecordReader rr = new CSVRecordReader(1,',');
//        rr.initialize(new FileSplit(inputFile));
//
//        Schema inputDataSchema = new Schema.Builder()
//                .addColumnString("ID")
//                .addColumnInteger("age")
//                .addColumnCategorical("job",
//                        Arrays.asList("admin.","unknown","services","management","technician","retired",
//                                "blue-collar","housemaid","self-employed","student","entrepreneur","unemployed"))
//                .addColumnCategorical("marital",Arrays.asList("married","divorced","single"))
//                .addColumnCategorical("education",Arrays.asList("unknown","primary","secondary","tertiary"))
//                .addColumnCategorical("default", Arrays.asList("yes","no"))
//                .addColumnInteger("balance")
//                .addColumnCategorical("housing",Arrays.asList("yes","no"))
//                .addColumnCategorical("loan",Arrays.asList("yes","no"))
//                .addColumnCategorical("contact", Arrays.asList("telephone","cellular","unknown"))
//                .addColumnInteger("day")
//                .addColumnCategorical("month",Arrays.asList("jan","feb","mar","apr","may","jun",
//                        "jul","aug","sep","oct","nov","dec"))
//                .addColumnInteger("duration")
//                .addColumnInteger("campaign")
//                .addColumnInteger("pdays")
//                .addColumnInteger("previous")
//                .addColumnCategorical("poutcome", Arrays.asList("unknown","success","failure","other"))
//                .addColumnCategorical("subscribed", Arrays.asList("yes","no"))
//                .build();
//
//        TransformProcess testtp = new TransformProcess.Builder(inputDataSchema)
//                .removeColumns("ID")
//                .transform(new ReplaceInvalidWithIntegerTransform("age", 0))
//                .categoricalToInteger("job","marital","education","default","housing","loan",
//                        "contact","month","poutcome","subscribed")
//                .filter(new FilterInvalidValues())
//                .build();
//
//        Schema outputSchema = testtp.getFinalSchema();
//        System.out.println("\n\n\nSchema after transforming data:");
//        System.out.println(outputSchema);
//
//
//        List<List<Writable>> trainData = new ArrayList<>();
//
//        while(rr.hasNext()){
//            trainData.add(rr.next());
//        }
//        rr.reset();
//
//        List<List<Writable>> processedData = LocalTransformExecutor.execute(trainData, testtp);
//        System.out.println(trainData.size());
//        System.out.println(processedData.size());
//
//        Nd4j.getEnvironment().allowHelpers(false);
//
//        INDArray transformedNDArray = RecordConverter.toMatrix(DataType.FLOAT, processedData);
//
//        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new MinMaxSerializerStrategy());
//        NormalizerMinMaxScaler scaler = normalizerSerializer.restore("C:\\Users\\Admin\\Desktop\\model\\normalizer.zip");
//        scaler.transform(transformedNDArray);
//
//        INDArray output = model.output(transformedNDArray);
//        System.out.println(output);
//
//        List<List<Writable>> outputCollections = RecordConverter.toRecords(output);
//
//        FileWriter fileWriter = new FileWriter("output.txt");
//
//        for (int i = 0; i < output.size(0); i++) {
//
//            if (output.getRow(i).getDouble(0) > output.getRow(i).getDouble(1)) {
//                fileWriter.write("no\n");
//            } else {
//                fileWriter.write("yes\n");
//            }
//        }
//        fileWriter.close();
//
//
//    }
//
//
//}
package ai0209.Q1;

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MinMaxSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class TermDepositInference {

    public static void main(String[] args) throws Exception, IOException {

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("C:\\Users\\Admin\\Desktop\\model\\termdeposit.zip", true);

        File testFilePath = new ClassPathResource("ai0209/Q1/test.csv").getFile();
        CSVRecordReader testCsvRR = new CSVRecordReader(1, ',');
        testCsvRR.initialize(new FileSplit(testFilePath));

        Schema testSchema = getTestSchema();

        TransformProcess testTP = new TransformProcess.Builder(testSchema)
                .categoricalToInteger("job", "marital", "education", "default", "housing",
                        "loan", "contact", "month", "poutcome")
                .filter(new FilterInvalidValues())
                .build();

        List<List<Writable>> oriData = new ArrayList<>();

        while (testCsvRR.hasNext()) {
            oriData.add(testCsvRR.next());
        }
        testCsvRR.reset();

        List<List<Writable>> transformedData = LocalTransformExecutor.execute(oriData, testTP);
        System.out.println(transformedData.size());

        Nd4j.getEnvironment().allowHelpers(false);

        INDArray transformedNDArray = RecordConverter.toMatrix(DataType.FLOAT, transformedData);

        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new MinMaxSerializerStrategy());
        NormalizerMinMaxScaler scaler = normalizerSerializer.restore("C:\\Users\\Admin\\Desktop\\model\\normalizer.zip");
        scaler.transform(transformedNDArray);

        INDArray output = model.output(transformedNDArray);
        System.out.println(output);

        List<List<Writable>> outputCollections = RecordConverter.toRecords(output);

        try (FileWriter fileWriter = new FileWriter("output.txt")) {

            for (int i = 0; i < output.size(0); i++) {

                if (output.getRow(i).getDouble(0) > output.getRow(i).getDouble(1)) {
                    fileWriter.write("no\n");
                } else {
                    fileWriter.write("yes\n");
                }
            }
            fileWriter.close();

        }


    }

    static Schema getTestSchema() {

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
                .build();
    }

}