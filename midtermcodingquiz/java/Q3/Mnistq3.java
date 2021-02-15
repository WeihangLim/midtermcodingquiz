package midtermcodingquiz.Q3;

import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
import javafx.scene.input.MouseButton;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.StrokeLineCap;
import javafx.stage.Stage;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;


//Reference: MnistImageSave
public class Mnistq3Save
{
    private static Logger log = LoggerFactory.getLogger(Mnistq3Save.class);

    public static void main(String[] args) throws Exception
    {
        int height = 28;
        int width = 28;
        int channels = 1;
        int seed = 123;

        Random randNumGen = new Random(seed);
        int batchSize = 128;
        int outputNum = 10;
        int numEpochs = 1;

        // DataSet Iterator
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true, seed);


        // Scale values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(mnistTrain);
        mnistTrain.setPreProcessor(scaler);

        // Build Neural Network

        log.info("Build Model");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs())
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(256)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .backpropType(BackpropType.Standard)
                .setInputType(InputType.convolutional(height,width,channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(10));


        log.info("TRAIN MODEL");
        for(int i = 0; i<numEpochs; i++)
        {
            model.fit(mnistTrain);
        }


        log.info("SAVE TRAINED MODEL");
        // Where to save model
        File locationToSave = new File(System.getProperty("java.io.tmpdir"), "/midtermcodingquiz/Q3/natural_images.zip");
        log.info(locationToSave.toString());

        // boolean save Updater
        boolean saveUpdater = false;

        // ModelSerializer needs modelname, saveUpdater, Location
        ModelSerializer.writeModel(model,locationToSave,saveUpdater);

        log.info("PROGRAM IS FINISHED. PLEASE CLOSE");

    }


}

//Reference: MnistImageLoad

public class Mnistq3Load
{
    private static Logger log = LoggerFactory.getLogger(Mnistq3Load.class);

    public static void main(String[] args) throws Exception
    {
        int height = 150;
        int width = 150;
        int channels = 24;


        File modelSave =  new File(System.getProperty("java.io.tmpdir"), "/midtermcodingquiz/Q3/natural_images.zip");

        if(!modelSave.exists())
        {
            System.out.println("Model not exist. Abort");
            return;
        }
        File imageToTest = new ClassPathResource("/midtermcodingquiz/Q3/imagetotest/0.jpg").getFile();

        // 1. Load saved model
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);

        // 2. Load an image for testing
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // 3. Get the image into an INDarray
        INDArray image = loader.asMatrix(imageToTest);


        //	Preprocessing to 0-1 or 0-255
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);




		//  Pass to the neural net for prediction

        INDArray output = model.output(image);
        log.info("Label:         " + Nd4j.argMax(output, 1));
        log.info("Probabilities: " + output.toString());

    }

}

public class Mnistq3Classifier extends Application {

    private static final Logger log = LoggerFactory.getLogger(Mnistq3Classifier.class);
    private static final int canvasWidth = 150;
    private static final int canvasHeight = 150;

    private static final int height = 28;
    private static final int width = 28;
    private static final int channels = 3; // single channel for grayscale images
    private static final int outputNum = 6; // 6 categories
    // 0 = buildings
    // 1 = forest
    // 2 = glacier
    // 3 = mountain
    // 4 = sea
    // 5 = street
    private static final int batchSize = 500;
    private static final int nEpochs = 5;
    private static final double learningRate = 0.001;
    private static MultiLayerNetwork model = null;

    private static final int seed = 1234;

    public static void main(String[] args) throws Exception
    {

    //  Create iterator using the batch size for one iteration
        log.info("Data load and vectorization...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false, seed);


    // 1. Model configuration

        log.info("Network configuration and training...");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Nesterovs(learningRate, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(200)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, 1))
                // InputType.convolutional for normal image
                .backpropType(BackpropType.Standard)
                .build();

        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // evaluation while training (the score should go down)
        for (int i = 0; i < nEpochs; i++) {
            model.fit(mnistTrain);

            log.info("Completed epoch {}", i);
            Evaluation eval = model.evaluate(mnistTest);
            log.info(eval.stats());
            mnistTrain.reset();
            mnistTest.reset();
        }

//    // 2. Test on image
//
//        launch();
//    }
//
//    @Override
//    public void start(Stage stage) throws Exception {
//        Canvas canvas = new Canvas(canvasWidth, canvasHeight);
//        GraphicsContext ctx = canvas.getGraphicsContext2D();
//
//        ImageView imgView = new ImageView();
//        imgView.setFitHeight(100);
//        imgView.setFitWidth(100);
//        ctx.setLineWidth(10);
//        ctx.setLineCap(StrokeLineCap.SQUARE);
//        Label lblResult = new Label();
//
//        HBox hbBottom = new HBox(10, imgView, lblResult);
//        hbBottom.setAlignment(Pos.CENTER);
//        VBox root = new VBox(5, canvas, hbBottom);
//        root.setAlignment(Pos.CENTER);
//
//        Scene scene = new Scene(root, 680, 300);
//        stage.setScene(scene);
//        stage.setTitle("Draw a digit and hit enter (right-click to clear)");
//        stage.setResizable(false);
//        stage.show();
//
//        canvas.setOnMousePressed(e -> {
//            ctx.setStroke(Color.WHITE);
//            ctx.beginPath();
//            ctx.moveTo(e.getX(), e.getY());
//            ctx.stroke();
//        });
//        canvas.setOnMouseDragged(e -> {
//            ctx.setStroke(Color.WHITE);
//            ctx.lineTo(e.getX(), e.getY());
//            ctx.stroke();
//        });
//        canvas.setOnMouseClicked(e -> {
//            if (e.getButton() == MouseButton.SECONDARY) {
//                clear(ctx);
//            }
//        });
//        canvas.setOnKeyReleased(e -> {
//            if (e.getCode() == KeyCode.ENTER) {
//                BufferedImage scaledImg = getScaledImage(canvas);
//                imgView.setImage(SwingFXUtils.toFXImage(scaledImg, null));
//                try {
//                    predictImage(scaledImg, lblResult);
//                } catch (Exception e1) {
//                    e1.printStackTrace();
//                }
//            }
//        });
//        clear(ctx);
//        canvas.requestFocus();
//    }
//
//    private void clear(GraphicsContext ctx) {
//        ctx.setFill(Color.BLACK);
//        ctx.fillRect(0, 0, 300, 300);
//    }
//
//    private BufferedImage getScaledImage(Canvas canvas) {
//        WritableImage writableImage = new WritableImage(canvasWidth, canvasHeight);
//        canvas.snapshot(null, writableImage);
//        Image tmp = SwingFXUtils.fromFXImage(writableImage, null).getScaledInstance(28, 28, Image.SCALE_SMOOTH);
//        BufferedImage scaledImg = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
//        Graphics graphics = scaledImg.getGraphics();
//        graphics.drawImage(tmp, 0, 0, null);
//        graphics.dispose();
//        return scaledImg;
//    }
//
//    private void predictImage(BufferedImage img, Label lbl) throws IOException {
//        NativeImageLoader loader = new NativeImageLoader(28, 28, 3, true);
//        INDArray image = loader.asRowVector(img).reshape(new int[]{1,784});
//        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
//        scaler.transform(image);
//        INDArray output = model.output(image);
//        lbl.setText("Prediction: " + model.predict(image)[0] + "\n " + output);
//    }
//
//}