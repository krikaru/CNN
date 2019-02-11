import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        Container container = new Container();

        SplitToRGB splitToRGB = new SplitToRGB(container);
        ConvoluationalLayer convoluationalLayer = new ConvoluationalLayer(container);
        SubsamplingLayer subsamplingLayer = new SubsamplingLayer(container);
        EnterMLP enterMLP = new EnterMLP(container);
        HiddenLayer hiddenLayer = new HiddenLayer(container);
        Output output = new Output(container);

        double error = 1;
        while (error > 0.05) {
            splitToRGB.start("19.jpg");
            convoluationalLayer.start();
            subsamplingLayer.start();
            enterMLP.start();
            hiddenLayer.start();
            error = output.start();
            System.out.println("error = " + error);
            output.editWeight();
            hiddenLayer.editWeight();
            enterMLP.editWeight();
            subsamplingLayer.editWeight();
            convoluationalLayer.editWeight();
        }

        //test
        splitToRGB.start("19.jpg");
        convoluationalLayer.start();
        subsamplingLayer.start();
        enterMLP.start();
        hiddenLayer.start();
        error = output.start();
        System.out.println("error = " + error);
        if (error < 0.05) {
            System.out.println("правильная картинка");
        } else {
            System.out.println("неправильная картинка");
        }
//
//
        splitToRGB.start("01.jpg");
        convoluationalLayer.start();
        subsamplingLayer.start();
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
