import numpy as np
import math

TITLE_FONT = ("Helvetica", "16")
WEIGHT_COLOR = "#666666"
WEIGHT_COLOR_ACTIVE = "#BB6666"
ACTIVE_WEIGHT_THRESHOLD = 1


def create_circle(x, y, r, ctx): #center coordinates, radius
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return ctx.create_oval(x0, y0, x1, y1)


def render_layer(canvas, neurons, top_y, bottom_y, x, neuron_size, labels=None):
    # Scale and normalize biases around 1 to represent useful node scale factors (ranging from ~0.3 - ~1.7)
    neurons *= 10
    neurons += 1
    coordinates = []
    padding = ((bottom_y - top_y) - (len(neurons) * neuron_size)) / (len(neurons) + 1)
    for i in range(len(neurons)):
        b = neurons[i]
        y = top_y + (i * neuron_size) + ((i+1) * padding) + (neuron_size / 2)
        create_circle(x, y, (neuron_size / 2) * b, canvas)
        if labels:
            canvas.create_text(x+50, y, text=labels[i], font=TITLE_FONT)
        coordinates.append((x, y))
    return coordinates


def render_rescale(data, magnitude=2):
    scale = max(np.max(data), -1 * np.min(data))
    return magnitude * data / scale


def render_weights(canvas, l1_positions, l2_positions, w, threshold=0.2, values=None):
    # Find indices of top threshold% weight values
    render_cap = int(len(w) * threshold)
    # Unravels the weights, create a new array of the sorted indices, and transforms the indices into 2d weight indices
    # Then stacks up the array, squeezes out the extra dim, and cuts off everything after the max render cap
    to_render = np.dstack(np.unravel_index(np.argsort(-w.ravel()), w.shape)).squeeze()[:render_cap]
    for point in to_render:
        l2 = point[1]  # layer 1 neuron index
        l1 = point[0]  # layer 2 neuron index
        l2_pos = l2_positions[l2]
        l1_pos = l1_positions[l1]
        weight = w[l1][l2]
        fill = WEIGHT_COLOR
        weight = abs(weight)
        np.set_printoptions(threshold=np.inf)
        # Render activation if l1 activation values are supplied
        if values is not None:
            value = values[l1]
            if value >= ACTIVE_WEIGHT_THRESHOLD:
                fill = WEIGHT_COLOR_ACTIVE
        canvas.create_line(*l1_pos, *l2_pos, width=weight, fill=fill)
