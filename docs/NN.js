var ActivationFunction;
(function (ActivationFunction) {
    ActivationFunction[ActivationFunction["SIGMOID"] = 0] = "SIGMOID";
    ActivationFunction[ActivationFunction["RELU"] = 1] = "RELU";
    ActivationFunction[ActivationFunction["SOFTMAX"] = 2] = "SOFTMAX";
})(ActivationFunction || (ActivationFunction = {}));
var LossFunction;
(function (LossFunction) {
    LossFunction[LossFunction["MEAN_SQUARED_ERROR"] = 0] = "MEAN_SQUARED_ERROR";
    LossFunction[LossFunction["CROSS_ENTROPY"] = 1] = "CROSS_ENTROPY";
})(LossFunction || (LossFunction = {}));
var Regularization;
(function (Regularization) {
    Regularization[Regularization["NONE"] = 0] = "NONE";
    Regularization[Regularization["L1"] = 1] = "L1";
    Regularization[Regularization["L2"] = 2] = "L2";
})(Regularization || (Regularization = {}));
class Perceptron {
    constructor(activationFunction = ActivationFunction.SIGMOID) {
        this.activationFunction = activationFunction;
        this.weights = [];
        this.bias = 0;
        this.value = 0;
        this.delta = 0; // Initialize delta
    }
    activate(x) {
        switch (this.activationFunction) {
            case ActivationFunction.RELU:
                return Math.max(0, x);
            case ActivationFunction.SOFTMAX:
                // Softmax is usually applied to the entire layer, not individual perceptrons
                return Math.exp(x);
            case ActivationFunction.SIGMOID:
            default:
                return 1 / (1 + Math.exp(-x));
        }
    }
    activateDerivative(x) {
        switch (this.activationFunction) {
            case ActivationFunction.RELU:
                return x > 0 ? 1 : 0;
            case ActivationFunction.SIGMOID:
            default:
                return x * (1 - x);
        }
    }
}
class NN {
    constructor(activationFunction = ActivationFunction.SIGMOID, lossFunction = LossFunction.MEAN_SQUARED_ERROR, regularization = Regularization.NONE, regularizationRate = 0.01, // Default value for regularization rate
    useSGD = false, useDropout = false, dropoutRate = 0.5) {
        this.inputs = [];
        this.hiddenLayers = [];
        this.outputs = [];
        this.activationFunction = activationFunction;
        this.lossFunction = lossFunction;
        this.regularization = regularization;
        this.regularizationRate = regularizationRate;
        this.useSGD = useSGD;
        this.useDropout = useDropout;
        this.dropoutRate = dropoutRate;
    }
    addInputLayer(size) {
        for (let i = 0; i < size; i++) {
            this.inputs.push(new Perceptron(this.activationFunction));
        }
    }
    addHiddenLayer(size) {
        const pastLayer = this.hiddenLayers.length == 0 ? this.inputs : this.hiddenLayers[this.hiddenLayers.length - 1];
        const layer = [];
        for (let i = 0; i < size; i++) {
            let perceptron = new Perceptron(this.activationFunction);
            layer.push(perceptron);
            //initialize weights and bias
            perceptron.bias = 0;
            for (let j = 0; j < pastLayer.length; j++) {
                perceptron.weights.push(0);
            }
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
            this.outputs.push(new Perceptron(this.activationFunction));
        }
        //initialize weights and bias
        let lastLayer = this.hiddenLayers.length == 0 ? this.inputs : this.hiddenLayers[this.hiddenLayers.length - 1];
        for (let perceptron of this.outputs) {
            perceptron.bias = 0;
            for (let j = 0; j < lastLayer.length; j++) {
                perceptron.weights.push(0);
            }
        }
    }
    giveInputs(inputs) {
        for (let i = 0; i < this.inputs.length; i++) {
            this.inputs[i].value = inputs[i];
        }
    }
    randomizeWeights() {
        for (let layer of this.hiddenLayers) {
            for (let perceptron of layer) {
                perceptron.bias = Math.random() * 2 - 1; // Randomize between -1 and 1
                for (let i = 0; i < perceptron.weights.length; i++) {
                    perceptron.weights[i] = Math.random() * 2 - 1; // Randomize between -1 and 1
                }
            }
        }
        for (let perceptron of this.outputs) {
            perceptron.bias = Math.random() * 2 - 1; // Randomize between -1 and 1
            for (let i = 0; i < perceptron.weights.length; i++) {
                perceptron.weights[i] = Math.random() * 2 - 1; // Randomize between -1 and 1
            }
        }
    }
    execute() {
        const layers = [this.inputs, ...this.hiddenLayers, this.outputs];
        for (let i = 1; i < layers.length; i++) {
            const prevLayer = layers[i - 1];
            const layer = layers[i];
            for (let perceptron of layer) {
                let sum = perceptron.bias;
                for (let j = 0; j < prevLayer.length; j++) {
                    sum += prevLayer[j].value * perceptron.weights[j];
                }
                perceptron.value = perceptron.activate(sum);
            }
            // Apply dropout if enabled
            if (this.useDropout && i < layers.length - 1) {
                for (let perceptron of layer) {
                    if (Math.random() < this.dropoutRate) {
                        perceptron.value = 0;
                    }
                }
            }
        }
        // Apply softmax to the output layer if using softmax activation
        if (this.activationFunction === ActivationFunction.SOFTMAX) {
            const outputLayer = this.outputs;
            const sumExp = outputLayer.reduce((sum, perceptron) => sum + perceptron.value, 0);
            for (let perceptron of outputLayer) {
                perceptron.value = perceptron.value / sumExp;
            }
        }
    }
    calculateLoss(expected, actual) {
        let loss;
        switch (this.lossFunction) {
            case LossFunction.CROSS_ENTROPY:
                loss = -expected.reduce((sum, exp, i) => sum + exp * Math.log(actual[i]), 0);
                break;
            case LossFunction.MEAN_SQUARED_ERROR:
            default:
                loss = expected.reduce((sum, exp, i) => sum + Math.pow(exp - actual[i], 2), 0) / expected.length;
                break;
        }
        // Add regularization term
        const regularization = this.hiddenLayers.flat().concat(this.outputs).reduce((sum, perceptron) => {
            return sum + perceptron.weights.reduce((wSum, weight) => {
                switch (this.regularization) {
                    case Regularization.L1:
                        return wSum + Math.abs(weight);
                    case Regularization.L2:
                        return wSum + Math.pow(weight, 2);
                    case Regularization.NONE:
                    default:
                        return wSum;
                }
            }, 0);
        }, 0);
        return loss + this.regularizationRate * regularization;
    }
    backpropagate(expected, actual) {
        // Calculate output layer deltas
        for (let i = 0; i < this.outputs.length; i++) {
            const output = this.outputs[i];
            const error = expected[i] - actual[i];
            output.delta = error * output.activateDerivative(output.value);
        }
        // Calculate hidden layer deltas
        for (let i = this.hiddenLayers.length - 1; i >= 0; i--) {
            const layer = this.hiddenLayers[i];
            const nextLayer = i === this.hiddenLayers.length - 1 ? this.outputs : this.hiddenLayers[i + 1];
            for (let j = 0; j < layer.length; j++) {
                const perceptron = layer[j];
                let error = 0;
                for (let k = 0; k < nextLayer.length; k++) {
                    error += nextLayer[k].delta * nextLayer[k].weights[j];
                }
                perceptron.delta = error * perceptron.activateDerivative(perceptron.value);
            }
        }
    }
    updateWeights(learningRate) {
        // Update weights for hidden layers
        for (let i = 0; i < this.hiddenLayers.length; i++) {
            const layer = this.hiddenLayers[i];
            const prevLayer = i === 0 ? this.inputs : this.hiddenLayers[i - 1];
            for (let j = 0; j < layer.length; j++) {
                const perceptron = layer[j];
                for (let k = 0; k < prevLayer.length; k++) {
                    perceptron.weights[k] += learningRate * perceptron.delta * prevLayer[k].value;
                    if (this.regularization === Regularization.L1) {
                        perceptron.weights[k] -= this.regularizationRate * Math.sign(perceptron.weights[k]);
                    }
                    else if (this.regularization === Regularization.L2) {
                        perceptron.weights[k] -= this.regularizationRate * perceptron.weights[k];
                    }
                }
                perceptron.bias += learningRate * perceptron.delta;
            }
        }
        // Update weights for output layer
        const lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length - 1];
        for (let i = 0; i < this.outputs.length; i++) {
            const perceptron = this.outputs[i];
            for (let j = 0; j < lastHiddenLayer.length; j++) {
                perceptron.weights[j] += learningRate * perceptron.delta * lastHiddenLayer[j].value;
                if (this.regularization === Regularization.L1) {
                    perceptron.weights[j] -= this.regularizationRate * Math.sign(perceptron.weights[j]);
                }
                else if (this.regularization === Regularization.L2) {
                    perceptron.weights[j] -= this.regularizationRate * perceptron.weights[j];
                }
            }
            perceptron.bias += learningRate * perceptron.delta;
        }
    }
    train(inputs, expected, learningRate) {
        if (this.useSGD) {
            // Stochastic Gradient Descent
            const randomIndex = Math.floor(Math.random() * inputs.length);
            this.giveInputs([inputs[randomIndex]]);
            this.execute();
            const actual = this.outputs.map(perceptron => perceptron.value);
            const loss = this.calculateLoss([expected[randomIndex]], actual);
            this.backpropagate([expected[randomIndex]], actual);
            this.updateWeights(learningRate);
        }
        else {
            // Normal Gradient Descent
            this.giveInputs(inputs);
            this.execute();
            const actual = this.outputs.map(perceptron => perceptron.value);
            const loss = this.calculateLoss(expected, actual);
            this.backpropagate(expected, actual);
            this.updateWeights(learningRate);
        }
    }
    getValue() {
        const perceptron = this.outputs.reduce((maxPerceptron, perceptron) => perceptron.value > maxPerceptron.value ? perceptron : maxPerceptron, this.outputs[0]);
        const label = this.outputs.indexOf(perceptron);
        return { perceptron, label };
    }
}
//# sourceMappingURL=NN.js.map