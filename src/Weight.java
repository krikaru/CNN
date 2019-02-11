public class Weight {

    private double value;
    private double gradient = 0;
    private double dW = 0;

    public Weight(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getGradient() {
        return gradient;
    }

    public void setGradient(double gradient) {
        this.gradient = gradient;
    }

    public double getdW() {
        return dW;
    }

    public void setdW(double dW) {
        this.dW = dW;
    }

    @Override
    public String toString() {
        return "Weight{" +
                "value=" + value +
                '}';
    }
}
