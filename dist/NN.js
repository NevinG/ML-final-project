var ActivationFunction;
(function (ActivationFunction) {
    ActivationFunction[ActivationFunction["SIGMOID"] = 0] = "SIGMOID";
    ActivationFunction[ActivationFunction["RELU"] = 1] = "RELU";
    ActivationFunction[ActivationFunction["SOFTMAX"] = 2] = "SOFTMAX";
})(ActivationFunction || (ActivationFunction = {}));
class Perceptron {
    constructor() {
        console.log("created Perceptron");
        this.activationFunction = ActivationFunction.SIGMOID;
        this.weights = [];
        this.bias = 0;
    }
}
class NN {
    constructor() {
        console.log("created NN");
        this.inputs = [];
        this.hiddenLayers = [];
        this.outputs = [];
    }
    addInputLayer(size) {
        for (let i = 0; i < size; i++) {
            this.inputs.push(new Perceptron());
        }
    }
    addHiddenLayer(size) {
        const layer = [];
        for (let i = 0; i < size; i++) {
            layer.push(new Perceptron());
        }
        this.hiddenLayers.push(layer);
    }
    addHiddenLayers(sizes) {
        for (let size of sizes) {
            this.addHiddenLayer(size);
        }
    }
    addOutputLayer(size) {
        for (let i = 0; i < size; i++) {
            this.outputs.push(new Perceptron());
        }
    }
}
//# sourceMappingURL=NN.js.map