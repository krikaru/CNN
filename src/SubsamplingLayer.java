import java.util.Arrays;
import java.util.List;

public class SubsamplingLayer {
     private List<Neuron[][]> subsList;
    private List<Neuron[][]> prevList;
    private boolean beforeMLP;

    private final double E = 0.0025;
    private final double A = 0.005;

    public SubsamplingLayer(List<Neuron[][]> subsList, List<Neuron[][]> prevList, boolean beforeMLP) {
        this.subsList = subsList;
        this.prevList = prevList;
        this.beforeMLP = beforeMLP;
    }

    public void start(){

        for (int count = 0; count < 6; count++){
            for (int row = 0; row < prevList.get(count).length; row = row + 2) {
                for (int column = 0; column < prevList.get(count)[0].length; column = column + 2) {

                    Neuron maxNeuron = prevList.get(count)[row][column];
                    double maxValue = maxNeuron.getValue();
                    int maxRow = row;
                    int maxColumn = column;
                    if (prevList.get(count)[row][column + 1].getValue() > maxValue) {
                        maxNeuron.setValue(0);
                        maxNeuron = prevList.get(count)[row][column + 1];
                        maxValue = maxNeuron.getValue();
                        maxRow = row;
                        maxColumn = column + 1;
                    } else {
                        prevList.get(count)[row][column + 1].setValue(0);
                    }
                    if (prevList.get(count)[row + 1][column].getValue() > maxValue) {
                        maxNeuron.setValue(0);
                        maxNeuron = prevList.get(count)[row + 1][column];
                        maxValue = maxNeuron.getValue();
                        maxRow = row + 1;
                        maxColumn = column;
                    } else {
                        prevList.get(count)[row + 1][column].setValue(0);
                    }
                    if (prevList.get(count)[row + 1][column + 1].getValue() > maxValue) {
                        maxNeuron.setValue(0);
                        maxNeuron = prevList.get(count)[row + 1][column + 1];
                        maxValue = maxNeuron.getValue();
                        maxRow = row + 1;
                        maxColumn = column + 1;
                    } else {
                        prevList.get(count)[row + 1][column + 1].setValue(0);
                    }

                    subsList.get(count)[row / 2][column / 2].setValue(maxValue);
                    subsList.get(count)[row / 2][column / 2].setRowMax(maxRow);
                    subsList.get(count)[row / 2][column / 2].setColumnMax(maxColumn);
                }
            }
        }
    }

    public void editWeight() {
        int maxWeight;
        double gradient;
        double dW;
        if (beforeMLP) {
            maxWeight = 1;
        } else {
            maxWeight = 25;
        }

        int iter = 0;
        for (Neuron[][] neuronsArr : subsList) {
            for (int i = 0; i < neuronsArr.length; i++)
                for (int j = 0; j < neuronsArr.length; j++) {
                    double sum = 0;

                    for (int countWeight = 0; countWeight < maxWeight; countWeight++) {
                        sum += neuronsArr[i][j].getWeightList().get(countWeight).getValue() * neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta();

                        gradient = neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta() * neuronsArr[i][j].getValue();
                        neuronsArr[i][j].getWeightList().get(countWeight).setGradient(gradient);

                        dW = E * gradient + A * neuronsArr[i][j].getWeightList().get(countWeight).getdW();
                        neuronsArr[i][j].getWeightList().get(countWeight).setdW(dW);

                        neuronsArr[i][j].getWeightList().get(countWeight).setValue(neuronsArr[i][j].getWeightList().get(countWeight).getValue() + dW);
                    }

                    if (beforeMLP) {
                        neuronsArr[i][j].setDelta(MathForCNN.derivativeHyperTan(neuronsArr[i][j].getValue()) * sum);
                    } else {
                        neuronsArr[i][j].setDelta(MathForCNN.derivitateLeakyReLU(neuronsArr[i][j].getValue()) * sum);
                    }

                    prevList.get(iter)[neuronsArr[i][j].getRowMax()][neuronsArr[i][j].getColumnMax()].setDelta(MathForCNN.derivativeHyperTan(neuronsArr[i][j].getValue()) * sum);
                }
            iter++;
        }
    }
}
