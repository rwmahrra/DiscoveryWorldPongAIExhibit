console.log("Hello world");
// Create a client instance
client = new Paho.MQTT.Client("localhost", 9001, "visualizer_module");

// set callback handlers
client.onConnectionLost = onConnectionLost;
client.onMessageArrived = onMessageArrived;

// connect the client
client.connect({onSuccess:onConnect});

// called when the client connects
function onConnect() {
  // Once a connection has been made, make a subscription and send a message.
  console.log("onConnect");
  client.subscribe("ai/structure");
  client.subscribe("ai/activation");
}

// called when the client loses its connection
function onConnectionLost(responseObject) {
  if (responseObject.errorCode !== 0) {
    console.log("onConnectionLost:"+responseObject.errorMessage);
  }
}

// called when a message arrives
function onMessageArrived(message) {
  last_activations = message.payloadString;
}

function render_game(ctx, frame, image_upscale = 5) {
    const canvas_width = document.body.clientWidth;
    const canvas_height = document.body.clientHeight;
    frame = scale(frame, 255);
    //Render game frame
    frame_width = 160 / 2; // Base state dimension, scaled down by two
    frame_height = 192 / 2;

    frameCanvas.width = frame_width
    frameCanvas.height = frame_height

    const frame_ctx = frameCanvas.getContext("2d");
    const imageData = frame_ctx.getImageData(0, 0, frame_width, frame_height);
    const frameData = imageData.data;
    for(let i = 0; i < frame.length; i++) {
        idx = i * 4;
        frameData[idx] = frame[i]; // Red
        frameData[idx+1] = frame[i]; // Green
        frameData[idx+2] = frame[i]; // Blue
        frameData[idx+3] = 255; // Alpha
    }

    frame_ctx.putImageData(imageData, 0, 0);

    img_w = frame_width * image_upscale;
    img_h = frame_height * image_upscale;
    img_x = canvas_width / 10;
    img_y = (canvas_height / 2) - (img_h / 2);

    ctx.drawImage(frameCanvas, img_x, img_y, img_w, img_h);
}

function base_render(ctx, frame) {
    /*
    Called after the first frame is rendered
    ctx: HTML5 canvas context
    frame: Pixels in initial game state
    */
    const canvas_width = document.body.clientWidth;
    const canvas_height = document.body.clientHeight;

    render_game(ctx, frame);

    pixel_dims = [frame_width, frame_height]

    // Store image-related rendering measurements
    img_coords = [img_x, img_y]
    img_size = [img_w, img_h]
    top_corner = [img_x, img_y]
    pixel_step = img_size[1] / pixel_dims[1]

    // Calculate positions of individual pixels, for use as a "layer" to connect edges
    pixel_pos = []
    for(let y = 0; y < pixel_dims[1]; y++) {
        for(let x = 0; x < pixel_dims[0]; x++) {
            x_pos = (x * pixel_step) + top_corner[0] + (pixel_step / 2)
            y_pos = (y * pixel_step) + top_corner[1] + (pixel_step / 2)
            pixel_pos.push([x_pos, y_pos])
        }
    }

    // Get neurons.
    // Structure: list of 2d and 1d arrays, [hidden weight, hidden bias, output weight, output bias]
    weights = example_structure
    hidden_biases = weights[1]
    output_biases = weights[3]

    // Normalize weights for use as line thickness
    output_weights = weights[2]
    hidden_weights = weights[0]

    //TODO: Implement
    const t1 = timer("partialSortHW");
    significant_hw = is_significant(hidden_weights, 0.02);
    t1.stop();
    const t2 = timer("partialSortOW");
    significant_ow = is_significant(output_weights, 0.3)
    t2.stop();

    // Determine appropriate base size for a neuron based on minimum allowable padding for a specific layer
    NEURON_SIZE = (canvas_height - (hidden_biases.length * MIN_PADDING)) / (hidden_biases.length)


    // Render neuron nodes, saving calculated positions for weight rendering
    // Hidden
    hidden_pos = render_layer(ctx, render_rescale(hidden_biases, magnitude=1), 0,
        canvas_height, HIDDEN_LAYER_X * canvas_width, NEURON_SIZE)

    // Output
    out_pos = render_layer(ctx, render_rescale(output_biases, magnitude=1), 0,
        canvas_height, OUTPUT_LAYER_X * canvas_width, NEURON_SIZE, null, OUTPUT_LABELS)

    // TODO: Create dynamic labels for inference confidence
    /*
    up_pos = out_pos[0]
    down_pos = out_pos[1]
    self.up_prob = tk.StringVar()
    self.down_prob = tk.StringVar()
    self.up_label = tk.Label(self.master, textvariable=self.up_prob, font=TITLE_FONT)
    self.down_label = tk.Label(self.master, textvariable=self.down_prob, font=TITLE_FONT)
    self.down_label.pack()
    self.up_label.pack()
    self.up_prob.set("0%")
    self.down_prob.set("0%")
    self.down_label.place(x=down_pos[0] + 25, y=down_pos[1] + 25)
    self.up_label.place(x=up_pos[0] + 25, y=up_pos[1] + 25)
    */
}

function render_tick(ctx, render_frame, state_frame, hl_activations, ol_activations) {
    const canvas_width = document.body.clientWidth;
    const canvas_height = document.body.clientHeight;

    // Clean slate before redraw
    ctx.clearRect(0, 0, canvas_width, canvas_height);

    // TODO: Update rendered probabilities
    /*
    percent_prob = prob * 100
    self.up_prob.set(f'{"{0:.2f}".format(percent_prob[0])}%')
    self.down_prob.set(f'{"{0:.2f}".format(percent_prob[1])}%')
    */
    let t = timer("render_game");
    // Render game frame
    render_game(ctx, render_frame);
    t.stop()

    // Render image weights
    t = timer("image_activations");
    const image_activations = is_firing(state_frame);
    const hw_activations = is_weight_active(hidden_weights, image_activations, copy(significant_hw));
    t.stop()

    t = timer("rescale hw")
    const rescaled_hw = render_rescale(hidden_weights);
    t.stop();
    t = timer("render_hidden_weights");
    render_weights(ctx, pixel_pos, hidden_pos, rescaled_hw, hw_activations)
    t.stop()

    // Re-compute hidden activations for rendering
    t = timer("hl_activations");
    ow_activations = is_weight_active(output_weights, hl_activations, copy(significant_ow))
    t.stop()

    // Render hidden weights
    t = timer("render_output_weights");
    render_weights(ctx, hidden_pos, out_pos, render_rescale(output_weights), ow_activations)
    t.stop()

    t = timer("render_layers");
    render_layer(ctx, render_rescale(hidden_biases, 1), 0,
        canvas_height, HIDDEN_LAYER_X * canvas_width, NEURON_SIZE, hl_activations)
    render_layer(ctx, render_rescale(output_biases, 1), 0,
        canvas_height, OUTPUT_LAYER_X * canvas_width, NEURON_SIZE, null, OUTPUT_LABELS, ol_activations)
    t.stop()
}

function render_loop() {
    if(last_activations && last_activations != last_rendered_activations) {
        const ctx = canvas.getContext("2d");
        const [state_frame, hl_activations, ol_activations] = JSON.parse(last_activations);
        render_tick(ctx, state_frame, state_frame, hl_activations, ol_activations);
        last_rendered_activations = last_activations;
    }
    requestAnimationFrame(render_loop);
}

function init() {
    canvas = document.getElementById("visualizer");

    // We will update this with game state pixels and embed it on the visualizer canvas
    frameCanvas = document.createElement('canvas');

    hidden_weights = example_structure[0];
    hidden_biases = example_structure[1];
    output_weights = example_structure[2];
    output_biases = example_structure[3];

    const ctx = canvas.getContext("2d");
    canvas.width = document.body.clientWidth;
    canvas.height = document.body.clientHeight;
    const { width, height } = canvas.getBoundingClientRect();
    const t = timer("base render");
    base_render(ctx, example_activation[0]);
    t.stop();
    const activations = [example_activation]//[activation_1, activation_2, activation_3, activation_4, activation_5];
    let i = 4;
    requestAnimationFrame(render_loop);
}

// Intentionally global. Canvas is for base drawing. Frame canvas holds game frame images.
var canvas = null;
var frameCanvas = null;

// Structure data
var hidden_weights = null;
var hidden_biases = null;
var output_weights = null;
var output_biases = null;

// Canvas rendering locations for nodes
var pixel_pos = null;
var hidden_pos = null;
var out_pos = null;

// Weights that are important enough to render
var significant_hw = null;
var significant_ow = null;

// Rendering config
var NEURON_SIZE = null;
var MIN_PADDING = 3;
var HIDDEN_LAYER_X = 0.5;
var OUTPUT_LAYER_X = 0.8;
var OUTPUT_LABELS = ["UP", "DOWN", "NONE"]

var last_activations = null;
var last_rendered_activations = null;


window.onload = init