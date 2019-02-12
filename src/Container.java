import java.util.ArrayList;
import java.util.List;

public class Container {
    private int kernelSize;
    private List<Neuron[][]> splitList = new ArrayList<>();
    private List<Weight[][]> betweenSplitAndConv = new ArrayList<>();

    private List<Neuron[][]> convList1 = new ArrayList<>();
    private List<Neuron[][]> subsList1 = new ArrayList<>();
    private List<Weight[][]> betweenSubs1AndConv2 = new ArrayList<>();

    private List<Neuron[][]> convList2 = new ArrayList<>();
    private List<Neuron[][]> subsList2 = new ArrayList<>();

    private List<Weight[][]> betweenSubs2AndConv3 = new ArrayList<>();

    private List<Neuron[][]> convList3 = new ArrayList<>();
    private List<Neuron[][]> subsList3 = new ArrayList<>();

    private List<Neuron> enterMLPList = new ArrayList<>();
    private List<Neuron> hiddenList = new ArrayList<>();
    private Neuron outputNeuron;


    Container(int kernelSize, int initialMatrixSize){
        this.kernelSize = kernelSize;
        containSplit(initialMatrixSize);
        containWeightAsArr(betweenSplitAndConv);

        //первая пара слоев
        int convSize = initialMatrixSize - kernelSize + 1;
//        System.out.println(convSize);
        if (isOdd(convSize)){
            containConv(convSize, convList1);
        }else {
            convSize = convSize + 1;
            containConv(convSize, convList1);
        }

        containSubs(convSize / 2, subsList1, false);

        containWeightAsArr(betweenSubs1AndConv2);

        //вторая пара слоев
        convSize = convSize/2 - kernelSize + 1;
//        System.out.println(convSize);

        if (isOdd(convSize)){
            containConv(convSize, convList2);
        }else {
            convSize = convSize + 1;
            containConv(convSize, convList2);
        }

        containSubs(convSize / 2, subsList2, false);

        containWeightAsArr(betweenSubs2AndConv3);

        //третья пара слоев
        convSize = convSize/2 - kernelSize + 1;
//        System.out.println(convSize);

        if (isOdd(convSize)){
            containConv(convSize, convList3);
        }else {
            convSize = convSize + 1;
            containConv(convSize, convList3);
        }

        containSubs(convSize / 2, subsList3, true);

        continEnterMLP();
        containHidden();
        containOutput();
    }

    private boolean isOdd(int value){
        return value % 2 == 0;
    }



    private void containSplit(int matrixSize){
        for (int count = 0; count < 6; count++){
            splitList.add(new Neuron[matrixSize][matrixSize]);

            for (int i = 0; i < matrixSize; i++){
                for (int j = 0; j < matrixSize; j++){
                    splitList.get(count)[i][j] = new Neuron(0);

                    for (int countWeight = 0; countWeight < Math.pow(kernelSize, 2); countWeight++) {

                        //init weight between this and next layer
                        splitList.get(count)[i][j].getWeightList().add(new Weight(0));
                        splitList.get(count)[i][j].getNeuronNext().add(new Neuron(0));
                    }
                }
            }
        }


    }

    private void containWeightAsArr(List<Weight[][]> weights){
        for (int count = 0; count < 6; count++){
            weights.add(new Weight[kernelSize][kernelSize]);

            for (int i = 0; i < kernelSize; i++) {
                for (int j = 0; j < kernelSize; j++) {
                    weights.get(count)[i][j] = new Weight((Math.random() * 1) - 0.5);
                }
            }
        }
    }

    private void containConv(int sizeMatrix, List<Neuron[][]> convList){
        for (int count = 0; count < 6; count++) {
            convList.add(new Neuron[sizeMatrix][sizeMatrix]);

            for (int i = 0; i < sizeMatrix; i++) {
                for (int j = 0; j < sizeMatrix; j++) {
                    convList.get(count)[i][j] = new Neuron(0);

                    for (int countWeight = 0; countWeight < Math.pow(kernelSize, 2); countWeight++) {
                        convList.get(count)[i][j].getPrevWeight().add(new Weight(0));
                        convList.get(count)[i][j].getNeuronPrev().add(new Neuron(0));
                    }
                }
            }
        }
    }

    /////////////////////////////////////////////////////////

    private void containSubs(int matrixSize, List<Neuron[][]> subsList, boolean beforeMLP){
        for (int count = 0; count < 6; count++) {
            subsList.add(new Neuron[matrixSize][matrixSize]);

            for (int i = 0; i < matrixSize; i++) {
                for (int j = 0; j < matrixSize; j++) {
                    subsList.get(count)[i][j] = new Neuron(0);

                    if (beforeMLP){
                        //только один некст нейрон
                        subsList.get(count)[i][j].getNeuronNext().add(new Neuron(0));
                        subsList.get(count)[i][j].getWeightList().add(new Weight(0));
                    }else {
                        for (int countWeight = 0; countWeight < 25; countWeight++){
                            subsList.get(count)[i][j].getNeuronNext().add(new Neuron(0));
                            subsList.get(count)[i][j].getWeightList().add(new Weight(0));
                        }

                    }


                }
            }
        }
    }

    private void continEnterMLP(){
        for (int count = 0; count < 6; count++) {
            enterMLPList.add(new Neuron(0));

            for (int weightCount = 0; weightCount < 12; weightCount++){
                enterMLPList.get(count).getWeightList().add(new Weight((Math.random() * 1) - 0.5));
            }

            for (int prevWeight = 0; prevWeight < Math.pow(subsList3.get(0).length, 2); prevWeight++){
                enterMLPList.get(count).getPrevWeight().add(new Weight((Math.random() * 1) - 0.5));
                enterMLPList.get(count).getNeuronPrev().add(new Neuron(0));
            }
        }
    }

    private void containHidden(){
        int i = 0;
        for (int count = 0; count < 24; count++) {
            hiddenList.add(new Neuron(0));
            hiddenList.get(count).getWeightList().add(new Weight((Math.random() * 1) - 0.5));

            if (count % 2 == 0){
                for (int prev = 0; prev < 6; prev = prev + 2){
                    hiddenList.get(count).getNeuronPrev().add(enterMLPList.get(prev));
                    hiddenList.get(count).getPrevWeight().add(enterMLPList.get(prev).getWeightList().get(i));
                }
            }else {
                for (int prev = 1; prev < 6; prev = prev + 2){
                    hiddenList.get(count).getNeuronPrev().add(enterMLPList.get(prev));
                    hiddenList.get(count).getPrevWeight().add(enterMLPList.get(prev).getWeightList().get(i));

                }
                i++;
            }
        }



        //        init in EnterMLP nextNeuron
        for (int count = 0; count < 6; count ++){
            if (count  % 2 == 0){
                for (int hiddenNeuron = 0; hiddenNeuron < 24; hiddenNeuron = hiddenNeuron + 2){
                    enterMLPList.get(count).getNeuronNext().add(hiddenList.get(hiddenNeuron));
                }
            }else {
                for (int hiddenNeuron = 1; hiddenNeuron < 24; hiddenNeuron = hiddenNeuron + 2){
                    enterMLPList.get(count).getNeuronNext().add(hiddenList.get(hiddenNeuron));
                }
            }
        }
    }

    public void containOutput(){
        outputNeuron = new Neuron(0);

        for (Neuron hiddenNeuron: hiddenList){
            outputNeuron.getNeuronPrev().add(hiddenNeuron);
            outputNeuron.getPrevWeight().add(hiddenNeuron.getWeightList().get(0));
        }

        for (int i = 0; i < 24; i++){
            hiddenList.get(i).getNeuronNext().add(outputNeuron);
        }
    }

    public List<Neuron[][]> getSplitList() {
        return splitList;
    }

    public List<Neuron[][]> getConvList1() {
        return convList1;
    }

    public List<Weight[][]> getBetweenSplitAndConv() {
        return betweenSplitAndConv;
    }

    public List<Neuron[][]> getSubsList1() {
        return subsList1;
    }

    public List<Neuron> getEnterMLPList() {
        return enterMLPList;
    }

    public List<Neuron> getHiddenList() {
        return hiddenList;
    }

    public Neuron getOutputNeuron() {
        return outputNeuron;
    }

    public List<Neuron[][]> getConvList2() {
        return convList2;
    }

    public List<Neuron[][]> getSubsList2() {
        return subsList2;
    }

    public List<Weight[][]> getBetweenSubs1AndConv2() {
        return betweenSubs1AndConv2;
    }

    public List<Weight[][]> getBetweenSubs2AndConv3() {
        return betweenSubs2AndConv3;
    }

    public List<Neuron[][]> getConvList3() {
        return convList3;
    }

    public List<Neuron[][]> getSubsList3() {
        return subsList3;
    }

    public int getKernelSize() {
        return kernelSize;
    }

}
