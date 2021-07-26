
//import {morph, opponent as op} from "/js/opponent.js";
// called when the client connects
function onConnect() {
  // Once a connection has been made, make a subscription and send a message.
  console.log("Connected to MQTT broker");
  client.subscribe("game/level");
  client.subscribe("player2/score"); // player 2 is ai
  client.subscribe("player1/score"); // player 1 is human
  client.subscribe("ai/activation");
  client.subscribe("depth/feed");
}

// called when the client loses its connection
function onConnectionLost(responseObject) {
  if (responseObject.errorCode !== 0) {
    console.log("onConnectionLost:"+responseObject.errorMessage);
  }
}

// // called when a message arrives
// function onMessageArrived(message) {
//     if(message.destinationName === "ai/activation") {
//         // Save the raw string instead of parsing because we may skip this
//         // frame if we get a new one before the next render - no need to waste cycles
//         last_activations = message.payloadString;
//     } else if(message.destinationName === "game/level") {
//         level = JSON.parse(message.payloadString)["level"];
//         model_initialized = false;
//         init_model(level);
//         console.log("next level");
//         //console.log(op);
//         //morph(op, morphTargets.Frustrated, 1)
//     }
// }
function myMethod(message) {
    //console.log(message.destinationName)
    if(message.destinationName === "game/level") {
        level = JSON.parse(message.payloadString)["level"];
        levelg = level
        model_initialized = false;
        //init_model(level);
        console.log("Changing to level:")
        console.log(level);
        ai_score = 0;
        player_score = 0;
        morphAllZero()
        morphOp("Smug", 0.5)
        clearInterval(rightInterval0);
        clearInterval(rightInterval1);
        clearInterval(leftInterval0);
        clearInterval(leftInterval1);

        const info_ctx = infoCanvas.getContext("2d");

        switch(level) {
            case 0:
                level = 1
                levelg = 1
                console.log('level 0. level set to:')
                console.log(level)
                render_info(info_ctx, 0, INFO_TEXT)
                
                init_model(level);
                break;
            case 1:
                console.log('level 1')
                morphOp("Happy",0.0)
                setTimeout(smugZero, 0)
                rightInterval1 = setInterval(right10, 2910)
                rightInterval0 = setInterval(rightZero, 3650)
                leftInterval1 = setInterval(left10, 2750)
                leftInterval0 = setInterval(leftZero, 3550)

                render_info(info_ctx, 1, INFO_TEXT)
                init_model(level);

                break;
            case 2:
                setTimeout(smug07, 500)
                morphOp("Happy",0.2)
                setTimeout(smugZero, 5000)
                //setTimeout(mad10, 5000)
                rightInterval1 = setInterval(right10, 3709)
                rightInterval0 = setInterval(rightZero, 4550)
                leftInterval1 = setInterval(left10, 3610)
                leftInterval0 = setInterval(leftZero, 4200)

                render_info(info_ctx, 2, INFO_TEXT)
                
                init_model(level);
                break;
            case 3:
                setTimeout(smug09, 5000)
                morphOp("Happy",0.4)
                morphOp("Mad", 0.3)
                //setTimeout(smugZero, 5000)
                //setTimeout(mad10, 5000)
                rightInterval1 = setInterval(right10, 5505)
                //rightInterval0 = setInterval(rightZero, 5600)
                rightInterval1 = setInterval(left10, 4910)
                //rightInterval0 = setInterval(leftZero, 6250)

                render_info(info_ctx, 3, INFO_TEXT)
                init_model(level);
                break;
        }

    } else if(message.destinationName === "ai/activation") {
        last_activations = message.payloadString;
        //morphOp("Sad",0.5)
    } else if(message.destinationName === "player2/score") {
        newScore = JSON.parse(message.payloadString)["score"]
        //console.log("score received")
        if (ai_score !== newScore) {
            //console.log("ai scored")
            ai_score = newScore;
            switch(levelg) {
                case 1:
                    morphOp("Happy", 1.0)
                    setTimeout(happyZero, 1000)
                    break;
                case 2:
                    morphOp("Smug", 1.0)
                    setTimeout(smug07, 1000)
                    break;
                case 3:
                    morphOp("Smug", 1.0)
                    setTimeout(smug09, 1000)
                    morphOp("Happy", 0.5)
                    setTimeout(happyZero, 1000)
                    break;
            }
        }
    } else if(message.destinationName === "player1/score") {
        newScore = JSON.parse(message.payloadString)["score"]
        //console.log("score received")
        if (player_score !== newScore) {
            //console.log("player scored")
            player_score = newScore;
            switch(levelg) {
                case 1:
                    morphOp("Sad", 1.0)
                    morphOp("Smug", 0.0)
                    setTimeout(sadZero, 1000)
                    break;
                case 2:
                    morphOp("Confused", 1.0)
                    morphOp("Smug", 0.0)
                    setTimeout(confusedZero, 1000)
                    setTimeout(smug07, 1000)
                    break;
                case 3:
                    morphOp("Confused", 1.0)
                    morphOp("Mad", 0.8)
                    morphOp("Smug", 0.0)
                    setTimeout(confusedZero, 1000)
                    setTimeout(mad03, 1000)
                    setTimeout(smug09, 1000)
                    break;
            }
        }
    } else if(message.destinationName === "depth/feed") {
        //console.log('received depth image')
        depthFeedStr = (JSON.parse(message.payloadString)["feed"])
        //console.log(depthFeedStr)
        //decoded = atob(rawfeed)

        
    } 
}

function smugZero() {
    morphOp("Smug", 0.0)
}
function confusedZero() {
    morphOp("Confused", 0.0)
}
function madZero() {
    morphOp("Mad", 0.0)
}
function mad03() {
    morphOp("Mad", 0.3)
}

function smug04() {
    morphOp("Smug", 0.4)
}
function smug07() {
    morphOp("Smug", 0.7)
}
function smug09() {
    morphOp("Smug", 0.9)
}
function mad10() {
    morphOp("Mad", 0.5)
    morphOp("Sad", 0.2)
}

function happyZero() {
    morphOp("Happy", 0.0)
}
function sadZero() {
    morphOp("Sad", 0.0)
}

function morphAllZero() {
    morphOp("Happy",0.2)
    morphOp("Frustrated",0.0)
    morphOp("Mad",0.0)
    morphOp("Smug",0.0)
    morphOp("Sad",0.0)
    morphOp("Right",0.0)
    morphOp("Left",0.0)
    morphOp("Confused",0.0)
}

function right10() {
    morphOp("Right", Math.random())
}
function rightZero() {
    morphOp("Right", 0.0)
}
function left10() {
    morphOp("Left", Math.random())
}
function leftZero() {
    morphOp("Left", 0.0)
}


function render_game(ctx, frame, image_upscale = 2) {
    frame = scale(frame, 255);

    frameCanvas.width = frame_width
    frameCanvas.height = frame_height
    //console.log(frameCanvas.x_pos)
    //frameCanvas.x_pos = 50;
    //console.log(frameCanvas.x_pos)

    const frame_ctx = frameCanvas.getContext("2d");
    const imageData = frame_ctx.getImageData(0, 0, frame_width, frame_height);
    const frameData = imageData.data;
    // frameData = imageData.data.map(function(element) {
    //     return element;
    // });
    //console.log(Math.max(frame))
    for(let i = 0; i < frame.length; i++) {
        idx = i * 4;
        // console.log("frame[i]  is:")
        // console.log(frame[i])
        //if (frame[i] !== 0) {console.log("frame has nonzero value")} 
        frameData[idx] = 230-frame[i]; // Red
        frameData[idx+1] = 230-frame[i]; // Green
        frameData[idx+2] = 230-frame[i]; // Blue
        frameData[idx+3] = 255; // Alpha
    }

    // img_w = frame_width * image_upscale;
    // img_h = frame_height * image_upscale;
    frame_ctx.putImageData(imageData, 0, 0);
    ctx.drawImage(frameCanvas, img_x, img_y, img_w, img_h); // LW

}
function render_depth_feed(ctx, image_upscale = 3.6) {
    var image = new Image();
    image.src = 'data:image/jpg;base64,' + depthFeedStr //canvas.toDataURL(depthFeedStr, 1)
    
    //depth_ctx.drawImage(image, 0, 0)
    image.onload = function() {

        // console.log('drawing image')
        ctx.drawImage(image, (d_canvas_width/3) - (image.width/2), d_canvas_height -(image.height*image_upscale), image.width * image_upscale, image.height * image_upscale)
    }
}

function render_weight_image(ctx, hl_activations, image_upscale = 5) {
    // Get the neuron with the strongest activity
    const top_neuron = argmax(hl_activations);

    // Select its weight with respect to each input pixel
    let frame = get_weight_map(hidden_weights, top_neuron)
    max_weight = max(frame)
    frame = scale(frame, 127/max_weight);
    frame = add(frame, 127)
    //Render game frame
    const frame_width = 192; // Base state dimension, scaled down by two
    const frame_height = 160; // LW was /2

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
    img_x = document.body.clientWidth/2//(canvas_width - img_w)/2;
    img_y = canvas_height - (img_h);//+ (canvas_height / 10)); // LW

    ctx.drawImage(weightImageCanvas, img_x * 2, img_y, img_w, img_h);
}

function render_tick(ctx, render_frame, state_frame, hl_activations, ol_activations, d_ctx) {
    // Clean slate before redraw
    ctx.clearRect(0, 0, canvas_width, canvas_height); 

    // Update rendered probabilities
    percent_prob = scale(ol_activations, 100)
    up_prob = parseFloat(percent_prob[0]).toFixed(2)+"%"
    down_prob = parseFloat(percent_prob[1]).toFixed(2)+"%"
    none_prob = parseFloat(percent_prob[2]).toFixed(2)+"%"

    if (percent_prob[0] > percent_prob[1] & percent_prob[0] > percent_prob[2]) {labelChosen = 0}
    else if (percent_prob[1] > percent_prob[2]) {labelChosen = 1}
    else {labelChosen = 2}

    let t = timer("render_game");
    // Render game frame
    render_depth_feed(d_ctx)
    render_game(ctx, render_frame);
    t.stop()

    t = timer("render_weight_image");
    // Render game frame
    //render_weight_image(ctx, hl_activations);
    t.stop()
    /*************************** */
    // Render image weights
    t = timer("image_activations");
    const image_activations = is_firing(state_frame);
    const hw_activations = is_weight_active(hidden_weights, image_activations, copy(significant_hw));
    t.stop()

    t = timer("render_hidden_weights");
    //testVar = 1;
    render_weights(ctx, pixel_pos, hidden_pos, rescaled_hw, hw_activations, 1)
    t.stop()
    /******************************************** */
    // Re-compute hidden activations for rendering
    t = timer("hl_activations");
    ow_activations = is_weight_active(output_weights, hl_activations, copy(significant_ow))
    t.stop()

    // Render hidden weights
    t = timer("render_output_weights");
    render_weights(ctx, hidden_pos, out_pos, render_rescale(output_weights), ow_activations, 2)
    t.stop()

    /********************************************* */
    //console.log('about to do the new activations')
    // Re-compute middle activations for rendering
    t = timer("il_activations");
    iw_activations = is_weight_active(hidden_weights, hl_activations, copy(significant_hw))
    t.stop()

    // Render middle weights
    t = timer("render_middle_weights");
    render_weights(ctx, hidden_pos, hidden_pos, rescaled_hw, iw_activations, 3)
    t.stop()
    /************************************************** */

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
    // ctx.fillText(up_prob, up_pos[0], up_pos[1]-40);
    // ctx.fillText(down_prob, down_pos[0], down_pos[1]-40);
    // ctx.fillText(none_prob, none_pos[0], none_pos[1]-40);
}

function render_loop() {
    //console.log("render_loop()")
    if(last_activations && last_activations != last_rendered_activations) {
        const ctx = canvas.getContext("2d");
        const d_ctx = d_canvas.getContext("2d");
        //const info_ctx = infoCanvas.getContext("2d");
        
        const [state_frame, hl_activations, ol_activations] = JSON.parse(last_activations);
        render_tick(ctx, state_frame, state_frame, hl_activations, ol_activations, d_ctx);
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
        console.log("!initialized")
        init_model(structure);
    } else {
        const ctx = canvas.getContext("2d");
        structure = null;
        console.log("rendering model of level:")
        console.log(level)
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
        console.log('MIN_PADDING is:')
        console.log(MIN_PADDING);
        // console.log('hidden_biases.length is:')
        // console.log(hidden_biases.length);
        console.log('canvas_height is:')
        console.log(canvas_height);
        NEURON_SIZE = (canvas_height - (hidden_biases.length * MIN_PADDING)) / (hidden_biases.length)

        // Render neuron nodes, saving calculated positions for weight rendering
        hidden_pos = render_layer(ctx, render_rescale(hidden_biases, 1), 0,
            canvas_width, HIDDEN_LAYER_Y * canvas_height , NEURON_SIZE)

        out_pos = render_layer(ctx, render_rescale(output_biases, 1), 0,
            canvas_width, OUTPUT_LAYER_Y * canvas_height, NEURON_SIZE, null, OUTPUT_LABELS)

        model_initialized = true;
        console.log("inside init model about to do requestAnimationFrame(render_loop)")
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
    client.onMessageArrived = myMethod; // onMessageArrived;

    // connect the client
    client.connect({onSuccess:onConnect});

    canvas = document.getElementById("visualizer");
    const ctx = canvas.getContext("2d");

    
    d_canvas = document.getElementById("depth");
    const d_ctx = d_canvas.getContext("2d");

    infoCanvas = document.getElementById("info");
    const info_ctx = infoCanvas.getContext("2d");

    // Size canvas to full screen
    //canvas.width = 10;
    canvas.width = document.body.clientWidth/(6/2.5); //document.body.clientWidth;
    canvas.height = document.body.clientHeight*(.9);// /2.2;//document.body.clientHeight; // LW
    canvas.y = 50
    canvas.x = 0 // does nothing
    canvas.style.left = (15) + (60) + 'px';//(document.body.clientWidth/6)+'px'; // LW
    canvas.style.top = (15) + 'px';
    canvas.style.position = 'absolute';
    console.log("canvas.left is ")
    console.log(canvas.left)

    d_canvas.width = document.body.clientWidth/(1); //document.body.clientWidth;
    d_canvas.height = canvas.height//document.body.clientHeight /2.5;//document.body.clientHeight; // LW
    d_canvas.style.left = (0)+'px'; // LW
    d_canvas.style.top = (15) + 'px';
    d_canvas.style.position = 'absolute';

    d_canvas_width = d_canvas.width
    d_canvas_height = d_canvas.height
    
    infoCanvas.width = document.body.clientWidth/(6/3); //document.body.clientWidth;
    infoCanvas.height = document.body.clientHeight /1;//document.body.clientHeight; // LW
    infoCanvas.style.left = (document.body.clientWidth/(7/3.5))+'px'; // LW
    infoCanvas.style.top = (45) + 'px';//(document.body.clientHeight/(7/1.5)) + 'px';
    infoCanvas.style.position = 'absolute';

    infoCanvas_width = infoCanvas.width
    infoCanvas_height = infoCanvas.height

    // Save canvas dimensions
    //canvas_width = 10;//
    canvas_width = canvas.width;//document.body.clientWidth/2; // LW
    //canvas_height = 10; //
    canvas_height = canvas.height;//document.body.clientHeight /2; // LW

    img_x = (canvas_width - img_w)/2 -(canvas_width/4);
    //img_x = (canvas_width - img_w)/2; // LW
    img_y = canvas_height - (img_h);// + (canvas_height / 10)); // LW
    
    // We will update this with game state pixels and embed it on the visualizer canvas
    frameCanvas = document.createElement('canvas');
    // This one will hold a model weight image overlay to see what the network is picking up on
    weightImageCanvas = document.createElement('canvas');

    initialized = true;
}
 
function morphOp(value){
    console.log("empty morphOp function")};



// Intentionally global. Canvas is for base drawing. Frame canvas holds game frame images.
var canvas = null;
var d_canvas = null;
var frameCanvas = null;
var weightImageCanvas = null;
var infoCanvas = null;

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
var MIN_PADDING = 3.8; // LW was 3. Used at line 212
var HIDDEN_LAYER_Y = 0.475; // LQ was 0.35
var OUTPUT_LAYER_Y = 0.12; // LW was 0.1
var OUTPUT_LABELS = ["LEFT", "RIGHT", "NONE"]
var INFO_TEXT = ["Can you beat the AI? \nStep in to play...\n \nArtificial Intelligence does more than just play games! \nAI can be used for:\n    Self-Driving cars        Medical Diagnosis\n    Language Translation    Virtual Assistants", 
"Hello human, I am an Artificial Intelligence. \nMy brain is a 'Neural Network' that learns by playing. \n \nMove your body side to side to control your paddle.\nSee if you can beat me!\n \nArtificial Intelligence does more than just play games! \nAI can be used for:\n    Self-Driving cars        Medical Diagnosis\n    Language Translation    Virtual Assistants", 
"This is what I see and my Neural Network. Each circle \nis like a neuron in a human brain. The neurons that \nlight up choose if I move my paddle left or right.\n \nI made it easy for you that round... \nCan you beat a model that was trained for more time?\nArtificial Intelligence does more than just play games! \nAI can be used for:\n    Self-Driving cars        Medical Diagnosis\n    Language Translation    Virtual Assistants", 
"When I was learning to play this game, I got rewards \nthat told me what actions were good and bad. I changed \nmy Neural Network to improve a little each time.\nThis is called Reinforcement Learning.\n \nCan you beat my hardest model? I've trained thousands of times \nto perfect this Neural Network.\nArtificial Intelligence does more than just play games! \nAI can be used for:\n    Self-Driving cars        Medical Diagnosis\n    Language Translation    Virtual Assistants"];
var image_upscale = 4;//4
var frame_width = 192 / 2; // Base state dimension, scaled down by two
var frame_height = 160 / 2; // LW was /2
var img_w = frame_width * image_upscale;
var img_h = frame_height * image_upscale;
var img_x = null; // These need to be computed based on canvas dimensions
var img_y = null;
var canvas_width = null;
var canvas_height = null;
var d_canvas_width = null;
var d_canvas_height = null;
var infoCanvas_width = null;
var infoCanvas_height = null;

var last_activations = null;
var last_rendered_activations = null;

let rightInterval1;
let rightInterval0;
let leftInterval1;
let leftInterval0;

let textInterval;

var levelg = 1;
var ai_score = 0;
var player_score = 0;

var labelChosen = 0;



var depthFeedStr = "";

window.onload = init