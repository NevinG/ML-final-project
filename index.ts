const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d");

//config elements
const layerNumInputs = document.getElementById("layer-num") as HTMLInputElement;
const hiddenLayerInputContainer = document.getElementById("hidden-layers-config") as HTMLDivElement;
const inputLayerInput = document.getElementById("input-layer") as HTMLDivElement;
const outputLayerInput = document.getElementById("output-layer") as HTMLDivElement;
const randomizeInputsButton = document.getElementById("randomize-inputs") as HTMLButtonElement;
const executeButton = document.getElementById("execute-nn") as HTMLButtonElement;
const randomizeWeights = document.getElementById("randomize-weights") as HTMLButtonElement;
const trainButton = document.getElementById("train-nn") as HTMLButtonElement;
const learningRateInput = document.getElementById("learning-rate") as HTMLInputElement;
const totalTrainingIterationsLabel = document.getElementById("training-iterations-label") as HTMLSpanElement;
const trainingProgressElement = document.getElementById("training-progress") as HTMLDivElement;
const trainingProgressContainerElement = document.getElementById("training-progress-container") as HTMLDivElement;
const randomTestInputButton = document.getElementById("random-test-input") as HTMLButtonElement;
const executeNNButton = document.getElementById("execute-nn") as HTMLButtonElement;
const executeNNAllButton = document.getElementById("execute-nn-all") as HTMLButtonElement;
const testingProgressElement = document.getElementById("testing-progress") as HTMLDivElement;
const testingProgressContainerElement = document.getElementById("testing-progress-container") as HTMLDivElement;
const trainingAccuracyLabel = document.getElementById("training-accuracy-label") as HTMLSpanElement;
const testingAccuracyLabel = document.getElementById("testing-accuracy-label") as HTMLSpanElement;
const trainOneIterationButton = document.getElementById("train-nn-all") as HTMLButtonElement;
const startTrainingButton = document.getElementById("start-training") as HTMLButtonElement;
const stopTrainingButton = document.getElementById("stop-training") as HTMLButtonElement;
//remove elements from dom to act as a variable
const hiddenLayerDiv = hiddenLayerInputContainer.children[0].cloneNode(true) as HTMLInputElement;
hiddenLayerInputContainer.removeChild(hiddenLayerInputContainer.children[0]);

//canvas setup
canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;

//config logic
layerNumInputs.onchange = (e) => {
  let val = parseInt((e.target as HTMLInputElement).value);
  renderHiddenLayerInputs(val);
  renderNNFromInputs();
}

//get data
//@ts-ignore
let trainData = mnist_handwritten_train as {image: number[], label: number}[]
//@ts-ignore
let testData = mnist_handwritten_test as {image: number[], label: number}[]
let currentImage: {image: number[], label: number}

//button logic
randomizeInputsButton.onclick = () => {
  const randomIndex = Math.floor(Math.random() * trainData.length);
  currentImage = trainData[randomIndex];
  nn.giveInputs(currentImage.image.map(x => x / 255));
  renderNN(nn);
}

executeButton.onclick = () => {
  nn.execute();
  renderNN(nn);
}

randomizeWeights.onclick = () => {
  nn.randomizeWeights();
  renderNN(nn);
}

trainButton.onclick = () => {
  const expected = Array(10).fill(0);
  expected[currentImage.label] = 1;

  const learningRate = parseFloat(learningRateInput.value);

  nn.train(currentImage.image.map(x => x / 255), expected, learningRate);
  renderNN(nn);
}

let dataIndex = 0;
let renderInterval: number;
let correctPredictions = 0;
let stopRequested = false;

trainOneIterationButton.onclick = () => {
  if(renderInterval != undefined) {
    console.log("ALREADY RUNNING CAN'T RUN IT AGAIN RN");
    return;
  }
  
  const learningRate = parseFloat(learningRateInput.value);
  dataIndex = 0;
  correctPredictions = 0;
  trainingProgressContainerElement.style.display = "block";
  renderInterval = setInterval(() => {
    const expected = Array(10).fill(0);
    expected[trainData[dataIndex].label] = 1;
    nn.train(trainData[dataIndex].image.map(x => x / 255), expected, learningRate);
    renderNN(nn);
    trainingProgressElement.style.width = `${(dataIndex / trainData.length) * 100}%`;

    const {perceptron, label} = nn.getValue();
    if (label === trainData[dataIndex].label) {
      correctPredictions++;
    }
    
    if(dataIndex == trainData.length - 1) {
      clearInterval(renderInterval);
      renderInterval = undefined;
      totalTrainingIterationsLabel.innerText = (parseInt(totalTrainingIterationsLabel.innerText) + 1).toString();
      trainingProgressContainerElement.style.display = "none";
      trainingAccuracyLabel.innerText = ((correctPredictions / trainData.length) * 100).toFixed(2) + "%";
    }

    dataIndex++;
  }, 0);
}

startTrainingButton.onclick = () => {
  if (renderInterval != undefined) {
    console.log("ALREADY RUNNING CAN'T RUN IT AGAIN RN");
    return;
  }

  startTrainingButton.disabled = true;
  stopTrainingButton.disabled = false;
  stopRequested = false;

  const learningRate = parseFloat(learningRateInput.value);
  dataIndex = 0;
  let correctPredictions = 0;
  trainingProgressContainerElement.style.display = "block";
  renderInterval = setInterval(() => {
    const expected = Array(10).fill(0);
    expected[trainData[dataIndex].label] = 1;
    nn.train(trainData[dataIndex].image.map(x => x / 255), expected, learningRate);
    renderNN(nn);
    trainingProgressElement.style.width = `${(dataIndex / trainData.length) * 100}%`;

    const { perceptron, label } = nn.getValue();
    if (label === trainData[dataIndex].label) {
      correctPredictions++;
    }

    if (dataIndex == trainData.length - 1) {
      totalTrainingIterationsLabel.innerText = (parseInt(totalTrainingIterationsLabel.innerText) + 1).toString();
      trainingAccuracyLabel.innerText = ((correctPredictions / trainData.length) * 100).toFixed(2) + "%";
      correctPredictions = 0;
    }

    dataIndex++;
    if (dataIndex >= trainData.length) {
      dataIndex = 0;
    }

    if (stopRequested && dataIndex === 0) {
      clearInterval(renderInterval);
      renderInterval = undefined;
      startTrainingButton.disabled = false;
      stopTrainingButton.disabled = true;
      trainingProgressContainerElement.style.display = "none";
    }
  }, 0);
}

stopTrainingButton.onclick = () => {
  stopRequested = true;
}

randomTestInputButton.onclick = () => {
  const randomIndex = Math.floor(Math.random() * testData.length);
  currentImage = testData[randomIndex];
  nn.giveInputs(currentImage.image.map(x => x / 255));
  renderNN(nn);
}

executeNNButton.onclick = () => {
  nn.execute();
  renderNN(nn);
}

let dataTestIndex = 0;
let renderTestInterval: number;
executeNNAllButton.onclick = () => {
  if(renderTestInterval != undefined) {
    console.log("ALREADY RUNNING CAN'T RUN IT AGAIN RN");
    return;
  }

  dataTestIndex = 0;
  let correctPredictions = 0;
  testingProgressContainerElement.style.display = "block";
  renderTestInterval = setInterval(() => {
    nn.giveInputs(testData[dataTestIndex].image.map(x => x / 255));
    nn.execute();
    renderNN(nn);
    testingProgressElement.style.width = `${(dataTestIndex / testData.length) * 100}%`;

    const {perceptron, label} = nn.getValue();
    if (label === testData[dataTestIndex].label) {
      correctPredictions++;
    }

    if (dataTestIndex === testData.length - 1) {
      clearInterval(renderTestInterval);
      renderTestInterval = undefined;
      testingProgressContainerElement.style.display = "none";
      testingAccuracyLabel.innerText = ((correctPredictions / testData.length) * 100).toFixed(2) + "%";
    }

    dataTestIndex++;
  }, 0);
}

//global var
let nn: NN;

const hiddenLayerNumbers = parseInt(layerNumInputs.value)
renderHiddenLayerInputs(hiddenLayerNumbers)
renderNNFromInputs()

function renderHiddenLayerInputs(val: number) {
  let curHiddenLayers = hiddenLayerInputContainer.children.length

  if (val > curHiddenLayers) {
    for (let i = curHiddenLayers; i < val; i++) {
      var hiddenLayer = hiddenLayerDiv.cloneNode(true);
      ((hiddenLayer as HTMLDivElement).children[1] as HTMLInputElement).onchange = renderNNFromInputs;
      hiddenLayerInputContainer.appendChild(hiddenLayer)
    }
  } else {
    for (let i = curHiddenLayers; i > val; i--) {
      hiddenLayerInputContainer.removeChild(hiddenLayerInputContainer.children[hiddenLayerInputContainer.children.length - 1]);
    }
  }

  //rename hidden layers to 'hidden i'
  for (let i = 0; i < hiddenLayerInputContainer.children.length; i++) {
    let hiddenLayer = hiddenLayerInputContainer.children[i].children[0] as HTMLInputElement;
    hiddenLayer.innerText = `HL ${i + 1}`
  }
}

function renderNNFromInputs () {
  nn = new NN();
  let inputLayerSize = parseInt((inputLayerInput.children[1] as HTMLInputElement).value);
  nn.addInputLayer(inputLayerSize);
  let hiddenLayers = [];
  for (let i = 0; i < hiddenLayerInputContainer.children.length; i++) {
    let hiddenLayer = hiddenLayerInputContainer.children[i].children[1] as HTMLInputElement;
    hiddenLayers.push(parseInt(hiddenLayer.value));
  }
  nn.addHiddenLayers(hiddenLayers);
  let outputLayerSize = parseInt((outputLayerInput.children[1] as HTMLInputElement).value);
  nn.addOutputLayer(outputLayerSize);
  renderNN(nn);
}

function renderNN(nn: NN) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const layers = [...nn.hiddenLayers, nn.outputs];

  const layerWidth = 50;
  const layerGap = 50;

  const leftPadding =
    (canvas.width - layers.length * (layerWidth + layerGap) - 320) / 2;

  
  let inputPixels: {x: number, y: number, value: number}[] = []

  //render inputs
  ctx.clearRect(0, 0, 280, canvas.height);
  for (let i = 0; i < nn.inputs.length; i++) {
    let x = i % 28;
    let y = Math.floor(i / 28);
    const topPadding = Math.floor((canvas.height - 280) / 2);
    //colors are inverted so we can see them well
    ctx.fillStyle = `rgb(${255 - nn.inputs[i].value * 255}, ${255 - nn.inputs[i].value * 255}, ${255 - nn.inputs[i].value * 255})`;
    ctx.fillRect(leftPadding + x * 10, y * 10 + topPadding, 10, 10);
    inputPixels.push({x: leftPadding + x * 10 + 5, y: y * 10 + topPadding + 5, value: nn.inputs[i].value});
  }

  //get layers
  let renderLayers: { perceptron: Perceptron; x: number; y: number; r: number }[][]= [];
  layers.forEach((layer, i) => {
    renderLayers.push(createLayers(layer, leftPadding + i * (layerWidth + layerGap), layerWidth));
  });

  //render connections
  renderConnections(renderLayers, inputPixels);

  //render perceptrons
  renderPerceptrons(renderLayers);

  //render inputs again so they are readable
  for (let i = 0; i < nn.inputs.length; i++) {
    let x = i % 28;
    let y = Math.floor(i / 28);
    const topPadding = Math.floor((canvas.height - 280) / 2);
    if(nn.inputs[i].value > 0) {
      ctx.fillStyle = `rgb(${255 - nn.inputs[i].value * 255}, ${255 - nn.inputs[i].value * 255}, ${255 - nn.inputs[i].value * 255})`;
      ctx.fillRect(leftPadding + x * 10, y * 10 + topPadding, 10, 10);
    }
  }
}

function createLayers(layer: Perceptron[], x: number, layerWidth: number) {
  const perceptronRadius = Math.min(layerWidth / 2, canvas.height / layer.length / 2);
  const perceptronGap = Math.min(50, (canvas.height - layer.length * perceptronRadius * 2 - 5) / (layer.length - 1));
  const topPadding = Math.max(0,(canvas.height - layer.length * (perceptronRadius * 2 + perceptronGap)) / 2)
  let res: { perceptron: Perceptron; x: number; y: number; r: number }[] = [];
  layer.forEach((perceptron, i) => {
    res.push({
      perceptron: perceptron,
      x: x + 320,
      y: perceptronRadius + topPadding + i * (perceptronRadius * 2 + perceptronGap),
      r: perceptronRadius,
    });
  });
  return res;
}

function renderConnections(layers: { perceptron: Perceptron; x: number; y: number; r: number }[][], inputPixels: {x: number, y: number, value: number}[]) {
  //connections to input
  for (let i = 0; i < inputPixels.length; i++) {
    for (let j = 0; j < layers[0].length; j++) {
      if (inputPixels[i].value > 0) {
        renderConnection(
          { perceptron: null, x: inputPixels[i].x, y: inputPixels[i].y, r: 0 },
          layers[0][j],
          layers[0][j].perceptron.weights[i] * 4 //so you can see them better
        );
      }
    }
  }

  //layer connections
  for (let i = 0; i < layers.length - 1; i++) {
    for (let j = 0; j < layers[i].length; j++) {
      for (let k = 0; k < layers[i + 1].length; k++) {
        renderConnection(layers[i][j], layers[i + 1][k], layers[i+1][k].perceptron.weights[j]);
      }
    }
  }
}

function renderConnection(perceptronA: { perceptron: Perceptron; x: number; y: number; r: number }, perceptronB: { perceptron: Perceptron; x: number; y: number; r: number }, weight: number) {
  ctx.beginPath();
  ctx.moveTo(perceptronA.x, perceptronA.y);
  ctx.lineTo(perceptronB.x, perceptronB.y);
  ctx.lineWidth = Math.max(.01,weight * 5); // Adjust the multiplier as needed for visibility
  ctx.stroke();
  ctx.lineWidth = 1; // Reset to default line width
}

function renderPerceptrons(layers: { perceptron: Perceptron; x: number; y: number; r: number }[][]) {
  const { perceptron: chosenPerceptron, label: predictedLabel } = nn.getValue();
  const actualLabel = currentImage?.label ?? -1;

  const isCorrect = predictedLabel === actualLabel;

  layers.forEach((layer, layerIndex) => {
    layer.forEach((perceptron, perceptronIndex) => {
      const rightLabel = layerIndex === layers.length - 1 ? perceptronIndex.toString() : undefined;
      let fillColor = { r: 255, g: 255, b: 255 };

      if (layerIndex === layers.length - 1 && chosenPerceptron === perceptron.perceptron) {
        fillColor = isCorrect ? { r: 0, g: 190, b: 0 } : { r: 255, g: 0, b: 0 };
      }

      renderPerceptron(perceptron.perceptron, perceptron.x, perceptron.y, perceptron.r, fillColor, rightLabel);
    });
  });
}

function renderPerceptron(
  perceptron: Perceptron,
  x: number,
  y: number,
  r: number,
  fillColor: {r: number, g: number, b: number} = {r: 255, g: 255, b: 255},
  rightLabel?: string
) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle = `rgb(${fillColor.r}, ${fillColor.g}, ${fillColor.b})`; // Set the fill color using fillColor
  ctx.fill();
  ctx.stroke();

  //in the middle of the circle write the value of the perceptron
  ctx.fillStyle = 'black';
  ctx.font = "20px Arial";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(perceptron.value?.toFixed(2) ?? "", x, y);

  if (rightLabel) {
    ctx.fillStyle = 'black';
    ctx.font = "16px Arial";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(rightLabel, x + r + 10, y);
  }
}
