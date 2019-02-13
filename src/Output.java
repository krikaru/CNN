public class Output {
     private Neuron outNeuron;

    public Output(Neuron outNeuron) {
        this.outNeuron = outNeuron;
    }

    public double start(){
        double sum = 0;
        for (int i = 0; i < 24; i++){
            sum += outNeuron.getNeuronPrev().get(i).getValue() * outNeuron.getPrevWeight().get(i).getValue();
        }
        outNeuron.setValue(MathForCNN.hyperTan(sum));

        return MathForCNN.rse(outNeuron.getValue());
    }

    public void editWeight(){
        double d0 = (1 - Math.abs(outNeuron.getValue())) * MathForCNN.derivativeHyperTan(outNeuron.getValue());
        outNeuron.setDelta(d0);
    }
}
