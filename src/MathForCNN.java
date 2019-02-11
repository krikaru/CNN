public class MathForCNN {

    public static double hyperTan(double value){
        return (Math.pow(Math.E, 2 * value) - 1)/(Math.pow(Math.E, 2 * value) + 1);
    }

    public static double leakyReLU(double value){
        if (value >= 0){
            return value;
        }else {
            return value * 0.1;
        }
    }

    public static double rse(double value){
        return Math.pow((1 - Math.abs(value)), 2);
    }

    public static double derivativeHyperTan(double value){
        return 1 - Math.pow(value, 2);
    }

    public static double derivitateLeakyReLU(double value){
        if (value >= 0){
            return 1;
        }else {
            return (Math.random() * 0.04) + 0.01;
        }
    }

    public static Neuron[][] rotMatrix180(Neuron[][] matrix){
        int dim = matrix.length;
        Neuron[][] arr1 = new Neuron[dim][dim];
        for(int row = 0; row < matrix[0].length; row++){
            for(int column = 0; column < matrix.length; column++){
                arr1[dim-1-row][dim-1-column] = matrix[row][column];
            }
        }
        return arr1;
    }


}
