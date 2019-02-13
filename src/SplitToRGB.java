import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class SplitToRGB implements Layer {
    private List<Neuron[][]> neuronList;
    private BufferedImage picture;

    private final double E = 0.00025;
    private final double A = 0.0005;

    public SplitToRGB(List<Neuron[][]> neuronList) {
        this.neuronList = neuronList;
    }

    public void start(String name) {
        try {
            picture = ImageIO.read(new File(name));
        } catch (IOException e) {
            e.printStackTrace();
        }
        start();
    }

    public void start() {
        int pixelR;
        int pixelG;
        int pixelB;

        for (int x = 0; x < picture.getWidth(); x++) {
            for (int y = 0; y < picture.getHeight(); y++) {
                final int rgb = picture.getRGB(x, y);
                pixelR = (rgb & 0xff0000) >> 16;
                pixelG = (rgb & 0xff00) >> 8;
                pixelB = rgb & 0xff;
                neuronList.get(0)[x][y].setValue((double)pixelR / 255);
                neuronList.get(1)[x][y].setValue((double)pixelR / 255);
                neuronList.get(2)[x][y].setValue((double)pixelG / 255);
                neuronList.get(3)[x][y].setValue((double)pixelG / 255);
                neuronList.get(4)[x][y].setValue((double)pixelB / 255);
                neuronList.get(5)[x][y].setValue((double)pixelB / 255);
            }
        }
    }

    public void editWeight(){

        double gradient = 0;
        double dW = 0;
        int matrixSize;

        for (Neuron[][] neuronsArr: neuronList){
            matrixSize = neuronsArr.length;
            for (int i = 0; i < matrixSize; i++) {
                for (int j = 0; j < matrixSize; j++) {
                    double sum = 0;
                    for (int countWeight = 0; countWeight < 25; countWeight++) {
                        sum += neuronsArr[i][j].getWeightList().get(countWeight).getValue() * neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta();

                        gradient = neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta() * neuronsArr[i][j].getValue();
                        neuronsArr[i][j].getWeightList().get(countWeight).setGradient(gradient);

                        dW = E * gradient + A * neuronsArr[i][j].getWeightList().get(countWeight).getdW();
                        neuronsArr[i][j].getWeightList().get(countWeight).setdW(dW);

                        neuronsArr[i][j].getWeightList().get(countWeight).setValue(neuronsArr[i][j].getWeightList().get(countWeight).getValue() + dW);
                    }
                    neuronsArr[i][j].setDelta(MathForCNN.derivitateLeakyReLU(neuronsArr[i][j].getValue()) * sum);
                }
            }
        }
    }
}
