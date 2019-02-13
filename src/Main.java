import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {

        System.out.println("init all collection");
        Container container = new Container(5, 90);

        SplitToRGB splitToRGB = new SplitToRGB(container.getSplitList());
        //первая пара
        ConvoluationalLayer convoluationalLayer1 =
                new ConvoluationalLayer(container.getConvList1(), container.getBetweenSplitAndConv(), container.getSplitList());
        SubsamplingLayer subsamplingLayer1 =
                new SubsamplingLayer(container.getSubsList1(), container.getConvList1(), false);

        //вторая пара
        ConvoluationalLayer convoluationalLayer2 =
                new ConvoluationalLayer(container.getConvList2(), container.getBetweenSubs1AndConv2(), container.getSubsList1());
        SubsamplingLayer subsamplingLayer2 =
                new SubsamplingLayer(container.getSubsList2(), container.getConvList2(), false);

        //вторая пара
        ConvoluationalLayer convoluationalLayer3 =
                new ConvoluationalLayer(container.getConvList3(), container.getBetweenSubs2AndConv3(), container.getSubsList2());
        SubsamplingLayer subsamplingLayer3 =
                new SubsamplingLayer(container.getSubsList3(), container.getConvList3(), true);

        //MLP layers
        EnterMLP enterMLP = new EnterMLP(container.getEnterMLPList(), container.getSubsList3());
        HiddenLayer hiddenLayer = new HiddenLayer(container.getHiddenList());
        Output output = new Output(container.getOutputNeuron());



        double error = 1;
        System.out.println("learn: 41.jpg ");
        while (error > 0.05) {
            splitToRGB.start("41.jpg");
            convoluationalLayer1.start();
            subsamplingLayer1.start();
            convoluationalLayer2.start();
            subsamplingLayer2.start();
            convoluationalLayer3.start();
            subsamplingLayer3.start();
            enterMLP.start();
            hiddenLayer.start();
            error = output.start();
//            System.out.println("error = " + error);
            output.editWeight();
            hiddenLayer.editWeight();
            enterMLP.editWeight();
            subsamplingLayer3.editWeight();
            subsamplingLayer2.editWeight();
            subsamplingLayer1.editWeight();
            splitToRGB.editWeight();
        }

//        test

        splitToRGB.start("32.jpg");
        convoluationalLayer1.start();
        subsamplingLayer1.start();
        convoluationalLayer2.start();
        subsamplingLayer2.start();
        convoluationalLayer3.start();
        subsamplingLayer3.start();
        enterMLP.start();
        hiddenLayer.start();
        error = output.start();

//        System.out.println("error = " + error);
        if (error < 0.05) {
            System.out.println("32.jpg правильная картинка");
        } else {
            System.out.println("32.jpg неправильная картинка");
        }

        splitToRGB.start("7.jpg");
        convoluationalLayer1.start();
        subsamplingLayer1.start();
        convoluationalLayer2.start();
        subsamplingLayer2.start();
        convoluationalLayer3.start();
        subsamplingLayer3.start();
        enterMLP.start();
        hiddenLayer.start();
        error = output.start();

//        System.out.println("error = " + error);
        if (error < 0.05) {
            System.out.println("7.jpg правильная картинка");
        } else {
            System.out.println("7.jpg неправильная картинка");
        }

        splitToRGB.start("41.jpg");
        convoluationalLayer1.start();
        subsamplingLayer1.start();
        convoluationalLayer2.start();
        subsamplingLayer2.start();
        convoluationalLayer3.start();
        subsamplingLayer3.start();
        enterMLP.start();
        hiddenLayer.start();
        error = output.start();

//        System.out.println("error = " + error);
        if (error < 0.05) {
            System.out.println("41.jpg правильная картинка");
        } else {
            System.out.println("41.jpg неправильная картинка");
        }

    }
}
