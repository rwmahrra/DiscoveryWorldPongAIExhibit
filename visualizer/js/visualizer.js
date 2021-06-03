// called when the client connects
function onConnect() {
  // Once a connection has been made, make a subscription and send a message.
  console.log("Connected to MQTT broker");
  client.subscribe("game/level");
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
    if(message.destinationName === "ai/activation") {
        // Save the raw string instead of parsing because we may skip this
        // frame if we get a new one before the next render - no need to waste cycles
        last_activations = message.payloadString;
    } else if(message.destinationName === "game/level") {
        level = JSON.parse(message.payloadString)["level"];
        model_initialized = false;
        init_model(level);
    }
}

function render_game(ctx, frame, image_upscale = 4) {
    frame = scale(frame, 255);

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

    ctx.drawImage(frameCanvas, img_x, img_y, img_w, img_h);
}

function render_weight_image(ctx, hl_activations, image_upscale = 4) {
    // Get the neuron with the strongest activity
    const top_neuron = argmax(hl_activations);

    // Select its weight with respect to each input pixel
    let frame = get_weight_map(hidden_weights, top_neuron)
    max_weight = max(frame)
    frame = scale(frame, 127/max_weight);
    frame = add(frame, 127)
    //Render game frame
    const frame_width = 192 / 2; // Base state dimension, scaled down by two
    const frame_height = 160 / 2;

    weightImageCanvas.width = frame_width
    weightImageCanvas.height = frame_height

    const frame_ctx = weightImageCanvas.getContext("2d");
    const imageData = frame_ctx.getImageData(0, 0, frame_width, frame_height);
    const frameData = imageData.data;
    for(let i = 0; i < frame.length; i++) {
        idx = i * 4;
        frameData[idx] = frame[i]; // Red
        frameData[idx+1] = frame[i]; // Green
        frameData[idx+2] = frame[i]; // Blue
        frameData[idx+3] = Math.max((frame[i] - 64), 0); // Alpha
    }

    frame_ctx.putImageData(imageData, 0, 0);

    img_w = frame_width * image_upscale;
    img_h = frame_height * image_upscale;
    img_x = (canvas_width - img_w)/2;
    img_y = canvas_height - (img_h + (canvas_height / 10));

    ctx.drawImage(weightImageCanvas, img_x, img_y, img_w, img_h);
}

function render_tick(ctx, render_frame, state_frame, hl_activations, ol_activations) {
    // Clean slate before redraw
    ctx.clearRect(0, 0, canvas_width, canvas_height);

    // Update rendered probabilities
    percent_prob = scale(ol_activations, 100)
    up_prob = parseFloat(percent_prob[0]).toFixed(2)+"%"
    down_prob = parseFloat(percent_prob[1]).toFixed(2)+"%"
    none_prob = parseFloat(percent_prob[2]).toFixed(2)+"%"

    let t = timer("render_game");
    // Render game frame
    render_game(ctx, render_frame);
    t.stop()

    t = timer("render_weight_image");
    // Render game frame
    //render_weight_image(ctx, hl_activations);
    t.stop()

    // Render image weights
    t = timer("image_activations");
    const image_activations = is_firing(state_frame);
    const hw_activations = is_weight_active(hidden_weights, image_activations, copy(significant_hw));
    t.stop()

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
        canvas_width, HIDDEN_LAYER_Y * canvas_height, NEURON_SIZE, hl_activations)
    render_layer(ctx, render_rescale(output_biases, 1), 0,
        canvas_width, OUTPUT_LAYER_Y * canvas_height, NEURON_SIZE, null, OUTPUT_LABELS, ol_activations)
    t.stop()

    // Create dynamic labels for inference confidence
    up_pos = out_pos[0]
    down_pos = out_pos[1]
    none_pos = out_pos[2]
    ctx.font = TITLE_FONT;
    ctx.textAlign = "center";
    ctx.fillText(up_prob, up_pos[0], up_pos[1]-40);
    ctx.fillText(down_prob, down_pos[0], down_pos[1]-40);
    ctx.fillText(none_prob, none_pos[0], none_pos[1]-40);
}

function render_loop() {
    if(last_activations && last_activations != last_rendered_activations) {
        const ctx = canvas.getContext("2d");
        const [state_frame, hl_activations, ol_activations] = JSON.parse(last_activations);
        render_tick(ctx, state_frame, state_frame, hl_activations, ol_activations);
        last_rendered_activations = last_activations;
    }
    // Break out of render loop if we receive a new model to render
    if(model_initialized) {
        requestAnimationFrame(render_loop);
    }
}

function init_model(level) {
    /*
    This initialization runs any time we receive a new model structure.
    It renders its weights and precomputes any convenient values for later use in realtime rendering.
    */
    if(!initialized) {
        // Busy wait if we receive a new model before the normal init is run.
        init_model(structure);
    } else {
        const ctx = canvas.getContext("2d");
        structure = null;
        switch(level) {
            case 1:
                structure = easy_model;
                break;
            case 2:
                structure = medium_model;
                break;
            case 3:
                structure = hard_model;
                break;
        }
        // Parse out the structure
        hidden_weights = structure[0];
        hidden_biases = structure[1];
        output_weights = structure[2];
        output_biases = structure[3];

        // Store image-related rendering measurements
        const pixel_dims = [frame_width, frame_height]
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

        rescaled_hw = render_rescale(hidden_weights);

        // Save the "significant" weights: we only render the 2% most important hidden weights and
        // the 30% most important output weights so the screen doesn't get crowded
        significant_hw = is_significant(hidden_weights, 0.02);
        significant_ow = is_significant(output_weights, 0.3)

        // Determine appropriate base size for a neuron based on minimum allowable padding for a specific layer
        NEURON_SIZE = (canvas_height - (hidden_biases.length * MIN_PADDING)) / (hidden_biases.length)

        // Render neuron nodes, saving calculated positions for weight rendering
        hidden_pos = render_layer(ctx, render_rescale(hidden_biases, 1), 0,
            canvas_width, HIDDEN_LAYER_Y * canvas_height, NEURON_SIZE)

        out_pos = render_layer(ctx, render_rescale(output_biases, 1), 0,
            canvas_width, OUTPUT_LAYER_Y * canvas_height, NEURON_SIZE, null, OUTPUT_LABELS)

        model_initialized = true;
        requestAnimationFrame(render_loop);
    }
}

function init() {
    /*
    Basic housekeeping initialization. Makes sure the canvases we need
    exist, and do any precomputing or sizing that aren't model dependent.
    */
    // Create a client instance
    client = new Paho.MQTT.Client("localhost", 9001, "visualizer_module");

    // set callback handlers
    client.onConnectionLost = onConnectionLost;
    client.onMessageArrived = onMessageArrived;

    // connect the client
    client.connect({onSuccess:onConnect});

    canvas = document.getElementById("visualizer");
    const ctx = canvas.getContext("2d");

    // Size canvas to full screen
    canvas.width = document.body.clientWidth;
    canvas.height = document.body.clientHeight;

    // Save canvas dimensions
    canvas_width = document.body.clientWidth;
    canvas_height = document.body.clientHeight;

    img_x = (canvas_width - img_w)/2;
    img_y = canvas_height - (img_h + (canvas_height / 10));

    // We will update this with game state pixels and embed it on the visualizer canvas
    frameCanvas = document.createElement('canvas');
    // This one will hold a model weight image overlay to see what the network is picking up on
    weightImageCanvas = document.createElement('canvas');

    initialized = true;
}

// Intentionally global. Canvas is for base drawing. Frame canvas holds game frame images.
var canvas = null;
var frameCanvas = null;
var weightImageCanvas = null;

// Track initialization status
var initialized = false;
var model_initialized = false;

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

// Precompute normalized hidden weights for nicer rendering
var rescaled_hw = null;

// Rendering config
var NEURON_SIZE = null;
var MIN_PADDING = 3;
var HIDDEN_LAYER_Y = 0.35;
var OUTPUT_LAYER_Y = 0.1;
var OUTPUT_LABELS = ["LEFT", "RIGHT", "NONE"]
var image_upscale = 4;
var frame_width = 192 / 2; // Base state dimension, scaled down by two
var frame_height = 160 / 2;
var img_w = frame_width * image_upscale;
var img_h = frame_height * image_upscale;
var img_x = null; // These need to be computed based on canvas dimensions
var img_y = null;
var canvas_width = null;
var canvas_height = null;

var last_activations = null;
var last_rendered_activations = null;


window.onload = init