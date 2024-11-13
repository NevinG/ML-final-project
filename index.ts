const canvas = document.getElementById('canvas') as HTMLCanvasElement
const ctx = canvas.getContext("2d")

//canvas setup
canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;

//create nn
const nn = new NN()
nn.addInputLayer(6)
nn.addHiddenLayers([5,4,3,2])
nn.addOutputLayer(1)

//display nn
renderNN(nn)


function renderNN(nn: NN) {
  const layers = [nn.inputs, ...nn.hiddenLayers, nn.outputs];

  const layerWidth = 50;
  const layerGap = 15;
  
  layers.forEach((layer, i) => {
    renderLayer(layer, layerWidth/2 + i * (layerWidth + layerGap), layerWidth)
  })
}

function renderLayer(layer: Perceptron[], x: number, layerWidth: number) {
  const perceptronGap = 5;
  const topPadding =  (canvas.height - layerWidth * layer.length ) / 2
  layer.forEach((perceptron, i) => {
    renderPerceptron(perceptron, x, topPadding + layerWidth / 2 + i * (50 + perceptronGap), layerWidth / 2)
  })
}

function renderPerceptron(perceptron: Perceptron, x: number, y: number, r: number) {
  ctx.beginPath()
  ctx.arc(x,y,r,0,2* Math.PI)
  ctx.stroke()
}
