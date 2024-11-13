enum ActivationFunction {
  SIGMOID,
  RELU,
  SOFTMAX,
}

class Perceptron {
  activationFunction: ActivationFunction;
  weights: number[];
  bias: number;

  constructor() {
    console.log("created Perceptron")
    this.activationFunction = ActivationFunction.SIGMOID
    this.weights = []
    this.bias = 0
  }
}

class NN {
  inputs: Perceptron[];
  hiddenLayers: Perceptron[][];
  outputs: Perceptron[];

  constructor() {
    console.log("created NN")
    this.inputs = []
    this.hiddenLayers = []
    this.outputs = []
  }

  addInputLayer(size: number) {
    for (let i = 0; i < size; i++) {
      this.inputs.push(new Perceptron())
    }
  }

  addHiddenLayer(size: number) {
    const layer = []
    for (let i = 0; i < size; i++) {
      layer.push(new Perceptron())
    }
    this.hiddenLayers.push(layer)
  }

  addHiddenLayers(sizes: number[]) {
    for(let size of sizes) {
      this.addHiddenLayer(size)
    }
  }

  addOutputLayer(size: number) {
    for (let i = 0; i < size; i++) {
      this.outputs.push(new Perceptron())
    }
  }
}

