import java.util.Arrays;
import java.util.List;

public class SubsamplingLayer {
     List<Neuron[][]> subsList;
    List<Neuron[][]> prevList;
    private boolean beforeMLP;

    private final double E = 0.00007;
    private final double A = 0.00003;

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
//                    System.out.println(row);
//                    System.out.println(column);
//                    System.out.println(prevNeurons.get(count)[row][column + 1]);
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
//        System.out.println("весов нет ----");
//        System.out.println("Нейроны на subsamling");
//        System.out.println(Arrays.deepToString(subsNeuron.get(0)));

    }

    public void editWeight(){
        int maxWeight;
        if (beforeMLP){
           maxWeight = 1;
        }else {
           maxWeight = 25;
        }

        int iter = 0;
        for (Neuron[][] neuronsArr:subsList){
            for (int i = 0; i < neuronsArr.length; i++) {
                for (int j = 0; j < neuronsArr.length; j++) {
                    double sum = 0;
                    double gradient = 0;
                    double dW = 0;


                    for (int countWeight = 0; countWeight < maxWeight; countWeight++) {
                        sum += neuronsArr[i][j].getWeightList().get(countWeight).getValue() * neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta();

                        gradient = neuronsArr[i][j].getNeuronNext().get(countWeight).getDelta() * neuronsArr[i][j].getValue();
                        neuronsArr[i][j].getWeightList().get(countWeight).setGradient(gradient);

                        dW = E * gradient + A * neuronsArr[i][j].getWeightList().get(countWeight).getdW();
                        neuronsArr[i][j].getWeightList().get(countWeight).setdW(dW);

                        neuronsArr[i][j].getWeightList().get(countWeight).setValue(neuronsArr[i][j].getWeightList().get(countWeight).getValue() + dW);
                    }

                    if (beforeMLP){
                        neuronsArr[i][j].setDelta(MathForCNN.derivativeHyperTan(neuronsArr[i][j].getValue()) * sum);
                    }else {
                        neuronsArr[i][j].setDelta(MathForCNN.derivitateLeakyReLU(neuronsArr[i][j].getValue()) * sum);
                    }

                    prevList.get(iter)[neuronsArr[i][j].getRowMax()][neuronsArr[i][j].getColumnMax()].setDelta(MathForCNN.derivativeHyperTan(neuronsArr[i][j].getValue()) * sum);
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
