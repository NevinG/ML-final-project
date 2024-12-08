const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
//config elements
const layerNumInputs = document.getElementById("layer-num");
const hiddenLayerInputContainer = document.getElementById("hidden-layers-config");
const inputLayerInput = document.getElementById("input-layer");
const outputLayerInput = document.getElementById("output-layer");
const randomizeInputsButton = document.getElementById("randomize-inputs");
const executeButton = document.getElementById("execute-nn");
const randomizeWeights = document.getElementById("randomize-weights");
const trainButton = document.getElementById("train-nn");
const learningRateInput = document.getElementById("learning-rate");
const totalTrainingIterationsLabel = document.getElementById("training-iterations-label");
const trainingProgressElement = document.getElementById("training-progress");
const trainingProgressContainerElement = document.getElementById("training-progress-container");
const randomTestInputButton = document.getElementById("random-test-input");
const executeNNButton = document.getElementById("execute-nn");
const executeNNAllButton = document.getElementById("execute-nn-all");
const testingProgressElement = document.getElementById("testing-progress");
const testingProgressContainerElement = document.getElementById("testing-progress-container");
const trainingAccuracyLabel = document.getElementById("training-accuracy-label");
const testingAccuracyLabel = document.getElementById("testing-accuracy-label");
const trainOneIterationButton = document.getElementById("train-nn-all");
const startTrainingButton = document.getElementById("start-training");
const stopTrainingButton = document.getElementById("stop-training");
const activationFunctionSelect = document.getElementById("activation-function");
const gradientDescentSelect = document.getElementById("gradient-descent");
const lossFunctionSelect = document.getElementById("loss-function");
const regularizationSelect = document.getElementById("regularization");
const regularizationRateInput = document.getElementById("regularization-rate");
const useDropoutCheckbox = document.getElementById("use-dropout");
const dropoutRateInput = document.getElementById("dropout-rate");
//remove elements from dom to act as a variable
const hiddenLayerDiv = hiddenLayerInputContainer.children[0].cloneNode(true);
hiddenLayerInputContainer.removeChild(hiddenLayerInputContainer.children[0]);
//canvas setup
canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;
//config logic
layerNumInputs.onchange = (e) => {
    let val = parseInt(e.target.value);
    renderHiddenLayerInputs(val);
    renderNNFromInputs();
};
//get data
//@ts-ignore
let trainData = mnist_handwritten_train;
//@ts-ignore
let testData = mnist_handwritten_test;
let currentImage;
//button logic
randomizeInputsButton.onclick = () => {
    const randomIndex = Math.floor(Math.random() * trainData.length);
    currentImage = trainData[randomIndex];
    nn.giveInputs(currentImage.image.map(x => x / 255));
    renderNN(nn);
};
executeButton.onclick = () => {
    nn.execute();
    renderNN(nn);
};
randomizeWeights.onclick = () => {
    nn.randomizeWeights();
    renderNN(nn);
};
trainButton.onclick = () => {
    const expected = Array(10).fill(0);
    expected[currentImage.label] = 1;
    const learningRate = parseFloat(learningRateInput.value);
    nn.train(currentImage.image.map(x => x / 255), expected, learningRate);
    renderNN(nn);
};
let dataIndex = 0;
let renderInterval;
let correctPredictions = 0;
let stopRequested = false;
trainOneIterationButton.onclick = () => {
    if (renderInterval != undefined) {
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
        const { perceptron, label } = nn.getValue();
        if (label === trainData[dataIndex].label) {
            correctPredictions++;
        }
        if (dataIndex == trainData.length - 1) {
            clearInterval(renderInterval);
            renderInterval = undefined;
            totalTrainingIterationsLabel.innerText = (parseInt(totalTrainingIterationsLabel.innerText) + 1).toString();
            trainingProgressContainerElement.style.display = "none";
            trainingAccuracyLabel.innerText = ((correctPredictions / trainData.length) * 100).toFixed(2) + "%";
        }
        dataIndex++;
    }, 0);
};
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
};
stopTrainingButton.onclick = () => {
    stopRequested = true;
    stopTrainingButton.disabled = true;
};
randomTestInputButton.onclick = () => {
    const randomIndex = Math.floor(Math.random() * testData.length);
    currentImage = testData[randomIndex];
    nn.giveInputs(currentImage.image.map(x => x / 255));
    renderNN(nn);
};
executeNNButton.onclick = () => {
    nn.execute();
    renderNN(nn);
};
let dataTestIndex = 0;
let renderTestInterval;
executeNNAllButton.onclick = () => {
    if (renderTestInterval != undefined) {
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
        const { perceptron, label } = nn.getValue();
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
};
//global var
let nn;
const hiddenLayerNumbers = parseInt(layerNumInputs.value);
renderHiddenLayerInputs(hiddenLayerNumbers);
renderNNFromInputs();
function renderHiddenLayerInputs(val) {
    let curHiddenLayers = hiddenLayerInputContainer.children.length;
    if (val > curHiddenLayers) {
        for (let i = curHiddenLayers; i < val; i++) {
            var hiddenLayer = hiddenLayerDiv.cloneNode(true);
            hiddenLayer.children[1].onchange = renderNNFromInputs;
            hiddenLayerInputContainer.appendChild(hiddenLayer);
        }
    }
    else {
        for (let i = curHiddenLayers; i > val; i--) {
            hiddenLayerInputContainer.removeChild(hiddenLayerInputContainer.children[hiddenLayerInputContainer.children.length - 1]);
        }
    }
    //rename hidden layers to 'hidden i'
    for (let i = 0; i < hiddenLayerInputContainer.children.length; i++) {
        let hiddenLayer = hiddenLayerInputContainer.children[i].children[0];
        hiddenLayer.innerText = `HL ${i + 1}`;
    }
}
function renderNNFromInputs() {
    const activationFunction = ActivationFunction[activationFunctionSelect.value];
    const useSGD = gradientDescentSelect.value === "sgd";
    const lossFunction = LossFunction[lossFunctionSelect.value];
    const regularization = Regularization[regularizationSelect.value];
    const regularizationRate = parseFloat(regularizationRateInput.value);
    const useDropout = useDropoutCheckbox.checked;
    const dropoutRate = parseFloat(dropoutRateInput.value);
    nn = new NN(activationFunction, lossFunction, regularization, regularizationRate, useSGD, useDropout, dropoutRate);
    let inputLayerSize = parseInt(inputLayerInput.children[1].value);
    nn.addInputLayer(inputLayerSize);
    let hiddenLayers = [];
    for (let i = 0; i < hiddenLayerInputContainer.children.length; i++) {
        let hiddenLayer = hiddenLayerInputContainer.children[i].children[1];
        hiddenLayers.push(parseInt(hiddenLayer.value));
    }
    nn.addHiddenLayers(hiddenLayers);
    let outputLayerSize = parseInt(outputLayerInput.children[1].value);
    nn.addOutputLayer(outputLayerSize);
    renderNN(nn);
    // Reset training and testing stats
    totalTrainingIterationsLabel.innerText = "0";
    trainingAccuracyLabel.innerText = "0%";
    testingAccuracyLabel.innerText = "0%";
}
function renderNN(nn) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const layers = [...nn.hiddenLayers, nn.outputs];
    const layerWidth = 50;
    const layerGap = 50;
    const leftPadding = (canvas.width - layers.length * (layerWidth + layerGap) - 320) / 2;
    let inputPixels = [];
    //render inputs
    ctx.clearRect(0, 0, 280, canvas.height);
    for (let i = 0; i < nn.inputs.length; i++) {
        let x = i % 28;
        let y = Math.floor(i / 28);
        const topPadding = Math.floor((canvas.height - 280) / 2);
        //colors are inverted so we can see them well
        ctx.fillStyle = `rgb(${255 - nn.inputs[i].value * 255}, ${255 - nn.inputs[i].value * 255}, ${255 - nn.inputs[i].value * 255})`;
        ctx.fillRect(leftPadding + x * 10, y * 10 + topPadding, 10, 10);
        inputPixels.push({ x: leftPadding + x * 10 + 5, y: y * 10 + topPadding + 5, value: nn.inputs[i].value });
    }
    //get layers
    let renderLayers = [];
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
        if (nn.inputs[i].value > 0) {
            ctx.fillStyle = `rgb(${255 - nn.inputs[i].value * 255}, ${255 - nn.inputs[i].value * 255}, ${255 - nn.inputs[i].value * 255})`;
            ctx.fillRect(leftPadding + x * 10, y * 10 + topPadding, 10, 10);
        }
    }
}
function createLayers(layer, x, layerWidth) {
    const perceptronRadius = Math.min(layerWidth / 2, canvas.height / layer.length / 2);
    const perceptronGap = Math.min(50, (canvas.height - layer.length * perceptronRadius * 2 - 5) / (layer.length - 1));
    const topPadding = Math.max(0, (canvas.height - layer.length * (perceptronRadius * 2 + perceptronGap)) / 2);
    let res = [];
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
function renderConnections(layers, inputPixels) {
    // Find the maximum absolute weight
    let maxWeight = 0;
    for (let i = 0; i < layers.length - 1; i++) {
        for (let j = 0; j < layers[i].length; j++) {
            for (let k = 0; k < layers[i + 1].length; k++) {
                maxWeight = Math.max(maxWeight, Math.abs(layers[i + 1][k].perceptron.weights[j]));
            }
        }
    }
    //connections to input
    for (let i = 0; i < inputPixels.length; i++) {
        for (let j = 0; j < layers[0].length; j++) {
            if (inputPixels[i].value > 0) {
                renderConnection({ perceptron: null, x: inputPixels[i].x, y: inputPixels[i].y, r: 0 }, layers[0][j], layers[0][j].perceptron.weights[i], maxWeight);
            }
        }
    }
    //layer connections
    for (let i = 0; i < layers.length - 1; i++) {
        for (let j = 0; j < layers[i].length; j++) {
            for (let k = 0; k < layers[i + 1].length; k++) {
                renderConnection(layers[i][j], layers[i + 1][k], layers[i + 1][k].perceptron.weights[j], maxWeight);
            }
        }
    }
}
function renderConnection(perceptronA, perceptronB, weight, maxWeight) {
    ctx.beginPath();
    ctx.moveTo(perceptronA.x, perceptronA.y);
    ctx.lineTo(perceptronB.x, perceptronB.y);
    const normalizedWeight = Math.abs(weight) / maxWeight;
    ctx.lineWidth = Math.max(.01, normalizedWeight * 0.6); // Normalize and set max thickness to 0.6 px
    // Calculate color based on weight
    let color;
    if (weight > 0.1) {
        const greenValue = Math.floor(255 * normalizedWeight);
        color = `rgba(0, ${greenValue}, 0, ${normalizedWeight})`; // More green for larger positive values with opacity
    }
    else if (weight < -0.1) {
        const redValue = Math.floor(255 * normalizedWeight);
        color = `rgba(${redValue}, 0, 0, ${normalizedWeight})`; // More red for larger negative values with opacity
    }
    else {
        color = `rgba(128, 128, 128, ${normalizedWeight})`; // Gray for zero weight with opacity
    }
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.lineWidth = 1; // Reset to default line width
    ctx.strokeStyle = 'black'; // Reset to default stroke color
}
function renderPerceptrons(layers) {
    var _a;
    const { perceptron: chosenPerceptron, label: predictedLabel } = nn.getValue();
    const actualLabel = (_a = currentImage === null || currentImage === void 0 ? void 0 : currentImage.label) !== null && _a !== void 0 ? _a : -1;
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
function renderPerceptron(perceptron, x, y, r, fillColor = { r: 255, g: 255, b: 255 }, rightLabel) {
    var _a, _b;
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
    ctx.fillText((_b = (_a = perceptron.value) === null || _a === void 0 ? void 0 : _a.toFixed(2)) !== null && _b !== void 0 ? _b : "", x, y);
    if (rightLabel) {
        ctx.fillStyle = 'black';
        ctx.font = "16px Arial";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(rightLabel, x + r + 10, y);
    }
}
function updateRegularizationRate() {
    regularizationRateInput.disabled = regularizationSelect.value === 'NONE';
}
function updateDropoutRate() {
    dropoutRateInput.disabled = !useDropoutCheckbox.checked;
}
regularizationSelect.addEventListener('change', updateRegularizationRate);
useDropoutCheckbox.addEventListener('change', updateDropoutRate);
updateRegularizationRate();
updateDropoutRate();
//# sourceMappingURL=index.js.map