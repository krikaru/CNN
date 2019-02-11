import java.util.ArrayList;
import java.util.List;

public class Container {
    private List<Neuron[][]> splitList = new ArrayList<>();
    private List<Neuron[][]> convList = new ArrayList<>();
    private List<Weight[][]> betweenSplitAndConv = new ArrayList<>();
    private List<Neuron[][]> subsList = new ArrayList<>();
    private List<Neuron> enterMLPList = new ArrayList<>();
    private List<Neuron> hiddenList = new ArrayList<>();
    private Neuron outputNeuron;


    Container(){
       containSplit();
       containConv();
       containWeightAsArr();
       containSubs();
       continEnterMLP();
       containHidden();
       containOutput();
    }



    private void containSplit(){
        for (int count = 0; count < 6; count++){
            splitList.add(new Neuron[10][10]);

            for (int i = 0; i < 10; i++){
                for (int j = 0; j < 10; j++){
                    splitList.get(count)[i][j] = new Neuron(0);

                    for (int countWeight = 0; countWeight < 9; countWeight++) {

                        //init weight between this and next layer
                        splitList.get(count)[i][j].getWeightList().add(new Weight(0));
                        splitList.get(count)[i][j].getNeuronNext().add(new Neuron(0));
                    }
                }
            }
        }


    }

    private void containConv(){
        for (int count = 0; count < 6; count++) {
            convList.add(new Neuron[8][8]);

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    convList.get(count)[i][j] = new Neuron(0);

                    for (int countWeight = 0; countWeight < 9; countWeight++) {
                        convList.get(count)[i][j].getPrevWeight().add(new Weight(0));
                        convList.get(count)[i][j].getNeuronPrev().add(new Neuron(0));
                    }
                }
            }
        }
    }

    private void containWeightAsArr(){
        for (int count = 0; count < 6; count++){
            betweenSplitAndConv.add(new Weight[3][3]);

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    betweenSplitAndConv.get(count)[i][j] = new Weight((Math.random() * 1) - 0.5);
                }
            }
        }
    }

    private void containSubs(){
        for (int count = 0; count < 6; count++) {
            subsList.add(new Neuron[4][4]);

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    subsList.get(count)[i][j] = new Neuron(0);

                    //только один некст нейрон
                    subsList.get(count)[i][j].getNeuronNext().add(new Neuron(0));
                    subsList.get(count)[i][j].getWeightList().add(new Weight(0));
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

            for (int prevWeight = 0; prevWeight < 16; prevWeight++){
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

    public List<Neuron[][]> getConvList() {
        return convList;
    }

    public List<Weight[][]> getBetweenSplitAndConv() {
        return betweenSplitAndConv;
    }

    public List<Neuron[][]> getSubsList() {
        return subsList;
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
}
