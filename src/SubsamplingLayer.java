import java.util.Arrays;
import java.util.List;

public class SubsamplingLayer {
    Container container;
    List<Neuron[][]> subsNeuron;
    List<Neuron[][]> convNeurons;

    private final double E = 0.07;
    private final double A = 0.03;

    public SubsamplingLayer(Container container) {
        this.container = container;
    }

    public void start(){
       subsNeuron = container.getSubsList();
       convNeurons = container.getConvList();

        for (int count = 0; count < 6; count++){
            for (int row = 0; row < convNeurons.get(count).length; row = row + 2) {
                for (int column = 0; column < convNeurons.get(count)[0].length; column = column + 2) {

                    Neuron maxNeuron = convNeurons.get(count)[row][column];
                    double maxValue = maxNeuron.getValue();
                    int maxRow = row;
                    int maxColumn = column;
//                    System.out.println(row);
//                    System.out.println(column);
//                    System.out.println(prevNeurons.get(count)[row][column + 1]);
                    if (Math.abs(convNeurons.get(count)[row][column + 1].getValue()) > Math.abs(maxValue)) {
                        maxNeuron.setValue(0);
                        maxNeuron = convNeurons.get(count)[row][column + 1];
                        maxValue = maxNeuron.getValue();
                        maxRow = row;
                        maxColumn = column + 1;
                    } else {
                        convNeurons.get(count)[row][column + 1].setValue(0);
                    }
                    if (Math.abs(convNeurons.get(count)[row + 1][column].getValue()) > Math.abs(maxValue)) {
                        maxNeuron.setValue(0);
                        maxNeuron =convNeurons.get(count)[row + 1][column];
                        maxValue = maxNeuron.getValue();
                        maxRow = row + 1;
                        maxColumn = column;
                    } else {
                        convNeurons.get(count)[row + 1][column].setValue(0);
                    }
                    if (Math.abs(convNeurons.get(count)[row + 1][column + 1].getValue()) > Math.abs(maxValue)) {
                        maxNeuron.setValue(0);
                        maxNeuron = convNeurons.get(count)[row + 1][column + 1];
                        maxValue = maxNeuron.getValue();
                        maxRow = row + 1;
                        maxColumn = column + 1;
                    } else {
                        convNeurons.get(count)[row + 1][column + 1].setValue(0);
                    }

                    subsNeuron.get(count)[row / 2][column / 2].setValue(maxValue);
                    subsNeuron.get(count)[row / 2][column / 2].setRowMax(maxRow);
                    subsNeuron.get(count)[row / 2][column / 2].setColumnMax(maxColumn);
                }
            }
        }
//        System.out.println("весов нет ----");
//        System.out.println("Нейроны на subsamling");
//        System.out.println(Arrays.deepToString(subsNeuron.get(0)));

    }

    public void editWeight(){
        int iter = 0;
        for (Neuron[][] neuronsArr:container.getSubsList()){
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    double sum = 0;
                    double gradient = 0;
                    double dW = 0;
                    for (int countWeight = 0; countWeight < 1; countWeight++) {
                        sum += neuronsArr[i][j].getWeightList().get(countWeight).getValue() * neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta();

                        gradient = neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta() * neuronsArr[i][j].getValue();
                        neuronsArr[i][j].getWeightList().get(countWeight).setGradient(gradient);

                        dW = E * gradient + A * neuronsArr[i][j].getWeightList().get(countWeight).getdW();
                        neuronsArr[i][j].getWeightList().get(countWeight).setdW(dW);

                        neuronsArr[i][j].getWeightList().get(countWeight).setValue(neuronsArr[i][j].getWeightList().get(countWeight).getValue() + dW);
                    }

                    neuronsArr[i][j].setDelta(MathForCNN.derivativeHyperTan(neuronsArr[i][j].getValue()) * sum);
                    container.getConvList().get(iter)[neuronsArr[i][j].getRowMax()][neuronsArr[i][j].getColumnMax()].setDelta(MathForCNN.derivativeHyperTan(neuronsArr[i][j].getValue()) * sum);
                }
            }
            iter++;
        }

//        System.out.println("----------- subs.editWeight------------");
//
//        System.out.println("Выход текущ нейрона:");
//        System.out.println(Arrays.deepToString(container.getSubsList().get(0)));
//
//        System.out.println("weight between subs and enter: 0 arr[0][0]");
//        System.out.println(container.getSubsList().get(0)[0][0].getWeightList());
//
//        System.out.println("enter's neuronNext: 0 arr [0][0] DELTA");
//        System.out.println(container.getSubsList().get(0)[0][0].getNeuronNext());
////
//        System.out.println("Delta subs neuron : 0 arr [0][0]");
//        System.out.println(container.getSubsList().get(0)[0][0].getDelta());
//////
//        System.out.println("gradient subs neuron : 0 neuron");
//        System.out.println(container.getSubsList().get(0)[0][0].getWeightList().get(0).getGradient());
//////
//        System.out.println("dW subs neuron: 0 neuron");
//        System.out.println(container.getSubsList().get(0)[0][0].getWeightList().get(0).getdW());
//////
//        System.out.println("New weight between subs and enter: 0 neuron");
//        System.out.println(container.getSubsList().get(0)[0][0].getWeightList());


//        for (int count = 0; count < 6; count++){
//            Neuron[][] neuronsArr = neuronList.get(count);
//            for (int i = 0; i < 2; i++){
//                for (int j = 0; j < 2; j++){
//                    double sum = 0;
//                    double gradient = 0;
//                    double dW = 0;
//                    for (int countWeight = 0; countWeight < 1; countWeight++){
//                        sum += neuronsArr[i][j].getWeightList().get(countWeight).getValue() * neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta();
//
//                        gradient = neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta() * neuronsArr[i][j].getValue();
//                        neuronsArr[i][j].getWeightList().get(countWeight).setGradient(gradient);
//
//                        dW = E * gradient + A * neuronsArr[i][j].getWeightList().get(countWeight).getdW();
//                        neuronsArr[i][j].getWeightList().get(countWeight).setdW(dW);
//
//                        neuronsArr[i][j].getWeightList().get(countWeight).setValue(neuronsArr[i][j].getWeightList().get(countWeight).getValue() + dW);
//                    }
//                    neuronsArr[i][j].setDelta(MathForCNN.derivativeHyperTan(neuronsArr[i][j].getValue()) * sum);
//                }
//            }
//        }
    }
}
