import tkinter as tk
import visualizer
from render_utils import render_weights, render_layer, render_rescale
import numpy as np
import cv2
from PIL import Image, ImageTk
from keras.models import Model

class RealtimeVisualizer:
    # Rendering constants
    MIN_PADDING = 3
    HIDDEN_LAYER_X = 1000
    OUTPUT_LAYER_X = 1600
    CANVAS_SIZE_720P = (1280, 720)
    CANVAS_SIZE_1080P = (1920, 1080)
    IMAGE_X = 400
    IMAGE_UPSCALE = 3
    OUTPUT_LABELS = ["UP", "DOWN"]

    def __init__(self, model):
        # Create canvas setup
        master = tk.Tk()
        self.canvas_width, self.canvas_height = self.CANVAS_SIZE_1080P
        self.ctx = tk.Canvas(master,
                   width=self.canvas_width,
                   height=self.canvas_height)
        self.ctx.pack()
        self.model = model

    def render_frame(self, frame):
        ctx = self.ctx
        # Render game frame
        pixel_dims = frame.shape
        frame = cv2.resize(frame, (frame.shape[0] * self.IMAGE_UPSCALE, frame.shape[1] * self.IMAGE_UPSCALE))
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        # Save rendering id of image to later inspect attributes
        img_id = ctx.create_image(self.IMAGE_X, self.canvas_height / 2, anchor="c", image=img)

        # Store image-related rendering measurements
        img_coords = ctx.coords(img_id)
        img_size = (img.width(), img.height())
        top_corner = (np.subtract(img_coords, np.divide(img_size, 2)))
        pixel_step = img_size[0] / pixel_dims[0]

        # Calculate positions of individual pixels, for use as a "layer" to connect edges
        pixel_pos = []
        for x in range(pixel_dims[0]):
            for y in range(pixel_dims[1]):
                x_pos = (x * pixel_step) + top_corner[0] + (pixel_step / 2)
                y_pos = (y * pixel_step) + top_corner[1] + (pixel_step / 2)
                pixel_pos.append((x_pos, y_pos))

        # Get neurons.
        # Structure: list of numpy arrays, [hidden weight, hidden bias, output weight, output bias]
        model = self.model
        weights = model.get_weights()
        hidden_biases = weights[1]
        output_biases = weights[3]

        # Normalize weights for use as line thickness
        output_weights = weights[2]
        hidden_weights = weights[0]

        # Determine appropriate base size for a neuron based on minimum allowable padding for a specific layer
        neuron_size = (self.canvas_height - (len(hidden_biases) * self.MIN_PADDING)) / len(hidden_biases)

        # Render neuron nodes, saving calculated positions for weight rendering
        hidden_pos = render_layer(ctx, render_rescale(hidden_biases, magnitude=1), 0, self.canvas_height, self.HIDDEN_LAYER_X, neuron_size)
        out_pos = render_layer(ctx, render_rescale(output_biases, magnitude=1), 0,
                               self.canvas_height, self.OUTPUT_LAYER_X, neuron_size, labels=self.OUTPUT_LABELS)
        # Render weights
        render_weights(ctx, hidden_pos, out_pos, render_rescale(output_weights), threshold=0.5)
        render_weights(ctx, pixel_pos, hidden_pos, render_rescale(hidden_weights), threshold=0.01, values=frame.ravel())

        # Create model for collecting hidden layer activations
        model = Model(inputs=model.inputs, outputs=model.layers[1].output)

        # Open window
        ctx.mainloop()
        tk.mainloop()
