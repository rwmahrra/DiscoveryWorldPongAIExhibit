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
        self.base_ctx = tk.Canvas(master,
                   width=self.canvas_width,
                   height=self.canvas_height)
        self.base_ctx.pack()
        self.model = model

    def base_render(self, frame):
        base_ctx = self.base_ctx
        frame[0, 79] = 1
        # Render game frame
        frame *= 256
        pixel_dims = frame.shape
        render_frame = cv2.resize(frame, (frame.shape[0] * self.IMAGE_UPSCALE, frame.shape[1] * self.IMAGE_UPSCALE))
        img = ImageTk.PhotoImage(image=Image.fromarray(render_frame))
        # Save rendering id of image to later inspect attributes
        img_id = self.base_ctx.create_image(self.IMAGE_X, self.canvas_height / 2, anchor="c", image=img)

        # Store image-related rendering measurements
        img_coords = base_ctx.coords(img_id)
        img_size = (img.width(), img.height())
        top_corner = (np.subtract(img_coords, np.divide(img_size, 2)))
        pixel_step = img_size[0] / pixel_dims[0]

        # Calculate positions of individual pixels, for use as a "layer" to connect edges
        self.pixel_pos = []
        for y in range(pixel_dims[0]):
            for x in range(pixel_dims[1]):
                x_pos = (x * pixel_step) + top_corner[0] + (pixel_step / 2)
                y_pos = (y * pixel_step) + top_corner[1] + (pixel_step / 2)
                self.pixel_pos.append((x_pos, y_pos))

        # Get neurons.
        # Structure: list of numpy arrays, [hidden weight, hidden bias, output weight, output bias]
        model = self.model
        weights = model.get_weights()
        self.hidden_biases = weights[1]
        self.output_biases = weights[3]

        # Normalize weights for use as line thickness
        self.output_weights = weights[2]
        self.hidden_weights = weights[0]

        # Determine appropriate base size for a neuron based on minimum allowable padding for a specific layer
        self.neuron_size = (self.canvas_height - (len(self.hidden_biases) * self.MIN_PADDING)) / len(self.hidden_biases)

        # Create model for collecting hidden layer activations
        self.hl_model = Model(inputs=self.model.inputs, outputs=self.model.layers[0].output)
        base_ctx.update()


    def render_frame(self, state_frame, render_frame, prob):
        # Render game frame
        render_frame *= 256
        pixel_dims = render_frame.shape
        render_frame = cv2.resize(render_frame, (render_frame.shape[0] * self.IMAGE_UPSCALE, render_frame.shape[1] * self.IMAGE_UPSCALE))
        img = ImageTk.PhotoImage(image=Image.fromarray(render_frame))
        self.base_ctx.create_image(self.IMAGE_X, self.canvas_height / 2, anchor="c", image=img)
        X = state_frame.reshape([1, state_frame.shape[0] * state_frame.shape[1]])
        hl_activations = self.hl_model.predict(X, batch_size=1).squeeze()
        # Render neuron nodes, saving calculated positions for weight rendering
        self.hidden_pos = render_layer(self.base_ctx, render_rescale(self.hidden_biases, magnitude=1), 0, self.canvas_height, self.HIDDEN_LAYER_X, self.neuron_size)
        self.out_pos = render_layer(self.base_ctx, render_rescale(self.output_biases, magnitude=1), 0,
                               self.canvas_height, self.OUTPUT_LAYER_X, self.neuron_size, labels=self.OUTPUT_LABELS)
        # Render weights
        render_weights(self.base_ctx, self.hidden_pos, self.out_pos, render_rescale(self.output_weights), threshold=0.5, values=hl_activations)
        render_weights(self.base_ctx, self.pixel_pos, self.hidden_pos, render_rescale(self.hidden_weights), threshold=0.01, values=render_frame.ravel())

        # Re-render labels
        render_layer(self.base_ctx, render_rescale(self.output_biases, magnitude=1), 0,
                     self.canvas_height, self.OUTPUT_LAYER_X, self.neuron_size, labels=[f'UP: {prob[0] * 100}%', f'DOWN: {prob[1] * 100}%'])
        # Open window
        self.base_ctx.update()
