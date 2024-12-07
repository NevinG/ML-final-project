enum ActivationFunction {
  SIGMOID,
  RELU,
  SOFTMAX,
}

class Perceptron {
  activationFunction: ActivationFunction;
  weights: number[];
  bias: number;
  value: number;
  delta: number; // Add delta for backpropagation

  constructor() {
    this.activationFunction = ActivationFunction.SIGMOID
    this.weights = []
    this.bias = 0
    this.value = 0
    this.delta = 0 // Initialize delta
  }
}

class NN {
  inputs: Perceptron[];
  hiddenLayers: Perceptron[][];
  outputs: Perceptron[];

  constructor() {
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
    const pastLayer = this.hiddenLayers.length == 0 ? this.inputs : this.hiddenLayers[this.hiddenLayers.length - 1]
    const layer = []
    for (let i = 0; i < size; i++) {
      let perceptron = new Perceptron()
      layer.push(perceptron)

      //initialize weights and bias
      perceptron.bias = 0
      for(let j = 0; j < pastLayer.length; j++) {
        perceptron.weights.push(0)
      }
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
    
    //initialize weights and bias
    let lastLayer = this.hiddenLayers.length == 0 ? this.inputs : this.hiddenLayers[this.hiddenLayers.length - 1]
    for(let perceptron of this.outputs) {
      perceptron.bias = 0
      for(let j = 0; j < lastLayer.length; j++) {
        perceptron.weights.push(0)
      }
    }
  }

  sigmoid(x: number) {
    return 1 / (1 + Math.exp(-x))
  }

  sigmoidDerivative(x: number) {
    return x * (1 - x)
  }

  giveInputs(inputs: number[]) {
    for (let i = 0; i < this.inputs.length; i++) {
      this.inputs[i].value = inputs[i]
    }
  }

  randomizeWeights() {
    for(let layer of this.hiddenLayers) {
      for(let perceptron of layer) {
        perceptron.bias = Math.random()
        for(let i = 0; i < perceptron.weights.length; i++) {
          perceptron.weights[i] = Math.random() / perceptron.weights.length //NOTE: talk about this in paper
        }
      }
    }

    for(let perceptron of this.outputs) {
      perceptron.bias = Math.random()
      for(let i = 0; i < perceptron.weights.length; i++) {
        perceptron.weights[i] = Math.random() / perceptron.weights.length
      }
    }
  }

  execute() {
    const layers = [this.inputs, ...this.hiddenLayers, this.outputs]
    for(let i = 1; i < layers.length; i++) {
      const prevLayer = layers[i - 1]
      const layer = layers[i]
      for(let perceptron of layer) {
        let sum = perceptron.bias
        for(let j = 0; j < prevLayer.length; j++) {
          sum += prevLayer[j].value * perceptron.weights[j]
        }
        perceptron.value = this.sigmoid(sum)
      }
    }
  }

  backpropagate(expected: number[]) {
    // Calculate output layer deltas
    for (let i = 0; i < this.outputs.length; i++) {
      const output = this.outputs[i];
      output.delta = (expected[i] - output.value) * this.sigmoidDerivative(output.value);
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
        perceptron.delta = error * this.sigmoidDerivative(perceptron.value);
      }
    }
  }

  updateWeights(learningRate: number) {
    // Update weights for hidden layers
    for (let i = 0; i < this.hiddenLayers.length; i++) {
      const layer = this.hiddenLayers[i];
      const prevLayer = i === 0 ? this.inputs : this.hiddenLayers[i - 1];
      for (let j = 0; j < layer.length; j++) {
        const perceptron = layer[j];
        for (let k = 0; k < prevLayer.length; k++) {
          perceptron.weights[k] += learningRate * perceptron.delta * prevLayer[k].value;
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
      }
      perceptron.bias += learningRate * perceptron.delta;
    }
  }

  train(inputs: number[], expected: number[], learningRate: number) {
    this.giveInputs(inputs);
    this.execute();
    this.backpropagate(expected);
    this.updateWeights(learningRate);
  }

  getValue() {
    const perceptron = this.outputs.reduce((maxPerceptron, perceptron) => perceptron.value > maxPerceptron.value ? perceptron : maxPerceptron, this.outputs[0])
    const label = this.outputs.indexOf(perceptron)
    return {perceptron, label}
  }
}

