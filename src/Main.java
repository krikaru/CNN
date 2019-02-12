import java.io.IOException;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) throws IOException {
        Container container = new Container(5, 90);

        SplitToRGB splitToRGB = new SplitToRGB(container.getSplitList());
        //первая пара
        ConvoluationalLayer convoluationalLayer1 = new ConvoluationalLayer(container.getConvList1(), container.getBetweenSplitAndConv(), container.getSplitList());
        SubsamplingLayer subsamplingLayer1 = new SubsamplingLayer(container.getSubsList1(), container.getConvList1(), false);

        //вторая пара
        ConvoluationalLayer convoluationalLayer2 = new ConvoluationalLayer(container.getConvList2(), container.getBetweenSubs1AndConv2(), container.getSubsList1());
        SubsamplingLayer subsamplingLayer2 = new SubsamplingLayer(container.getSubsList2(), container.getConvList2(), false);

        //вторая пара
        ConvoluationalLayer convoluationalLayer3 = new ConvoluationalLayer(container.getConvList3(), container.getBetweenSubs2AndConv3(), container.getSubsList2());
        SubsamplingLayer subsamplingLayer3 = new SubsamplingLayer(container.getSubsList3(), container.getConvList3(), true);

        EnterMLP enterMLP = new EnterMLP(container.getEnterMLPList(), container.getSubsList3());
        HiddenLayer hiddenLayer = new HiddenLayer(container.getHiddenList());
        Output output = new Output(container.getOutputNeuron());

        double error = 1;
        while (error > 0.05) {
            splitToRGB.start("41.jpg");
//        System.out.println(Arrays.deepToString(splitToRGB.getNeuronList().get(0)));
            convoluationalLayer1.start();
//        System.out.println(Arrays.deepToString(convoluationalLayer1.convList.get(0)));
            subsamplingLayer1.start();
//        System.out.println("---------------------------------------------");
//        System.out.println(Arrays.deepToString(subsamplingLayer1.subsList.get(0)));
            convoluationalLayer2.start();
//        System.out.println(Arrays.deepToString(convoluationalLayer2.convList.get(0)));
            subsamplingLayer2.start();
//        System.out.println("---------------------------------------------");
//        System.out.println(Arrays.deepToString(subsamplingLayer2.subsList.get(0)));
            convoluationalLayer3.start();
//            System.out.println(Arrays.deepToString(convoluationalLayer3.convList.get(0)));
            subsamplingLayer3.start();
//        System.out.println(Arrays.deepToString(subsamplingLayer3.subsList.get(0)));
//        System.out.println("---------------------------------------------");
//        System.out.println(Arrays.deepToString(subsamplingLayer3.prevList.get(0)));
            enterMLP.start();
//        System.out.println(enterMLP.enterList);
            hiddenLayer.start();
//        System.out.println(hiddenLayer.hiddenList);
            error = output.start();
//        System.out.println(output.outNeuron);
            System.out.println("error = " + error);
            output.editWeight();
//        System.out.println(output.outNeuron);
            hiddenLayer.editWeight();
//        System.out.println(hiddenLayer.hiddenList);
            enterMLP.editWeight();
//        System.out.println(enterMLP.enterList);
            subsamplingLayer3.editWeight();
//        System.out.println("---------------------------------------------");
//        System.out.println(Arrays.deepToString(subsamplingLayer3.subsList.get(0)));
            subsamplingLayer2.editWeight();
//        System.out.println(Arrays.deepToString(subsamplingLayer2.subsList.get(0)));
            subsamplingLayer1.editWeight();
//        System.out.println(Arrays.deepToString(subsamplingLayer1.subsList.get(0)));
            splitToRGB.editWeight();
        }

//        test
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
        System.out.println("error = " + error);
        if (error < 0.05) {
            System.out.println("правильная картинка");
        } else {
            System.out.println("неправильная картинка");
        }

        splitToRGB.start("Aaron_Eckhart_0001.jpg");
        convoluationalLayer1.start();
        subsamplingLayer1.start();
        convoluationalLayer2.start();
        subsamplingLayer2.start();
        convoluationalLayer3.start();
        subsamplingLayer3.start();
        enterMLP.start();
        hiddenLayer.start();
        error = output.start();
        System.out.println("error = " + error);
        if (error < 0.05) {
            System.out.println("правильная картинка");
        } else {
            System.out.println("неправильная картинка");
        }
    }
}
