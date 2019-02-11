public class Output {
    Container container;
    Neuron outNeuron;

    public Output(Container container) {
        this.container = container;
    }

    public double start(){
        outNeuron = container.getOutputNeuron();
        double sum = 0;
//        System.out.println("Веса между hidden и output:");
//        System.out.println(outNeuron.getPrevWeight());
        for (int i = 0; i < 24; i++){
            sum += outNeuron.getNeuronPrev().get(i).getValue() * outNeuron.getPrevWeight().get(i).getValue();

        }
        outNeuron.setValue(MathForCNN.hyperTan(sum));
//        System.out.println("outNeuron:");
//        System.out.println(outNeuron.getValue());
        double localError = MathForCNN.rse(outNeuron.getValue());
//        System.out.println("localError = " + localError);
        return localError;
    }

    public void editWeight(){
        double d0 = (1 - Math.abs(outNeuron.getValue())) * MathForCNN.derivativeHyperTan(outNeuron.getValue());
        outNeuron.setDelta(d0);
//        System.out.println("delta out neuron");
//        System.out.println(d0);
    }
}
