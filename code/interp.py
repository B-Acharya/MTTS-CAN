import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
from scipy.interpolate import interp1d

import tensorflow as tf


def make_gradcam_heatmap(inputs, model, last_conv_layer_name):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the first output (as used for HR prediction) for each input image
    # with respect to the activations of the last conv layer      
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(inputs)
        pulse_pred = preds[:, 0]    # first output per input image 

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(pulse_pred, last_conv_layer_output)

    # Calculate the mean intensity of the gradient for each the feature map in each input image 
    # (1400 images x 64 feature maps)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # for each image input multiply each channel in the feature map array
    # by "how important this channel is" with regard to the prediction (via pooled gradients)
    # then sum all the channels to obtain the final activation heatmap
    heatmaps = []
    for out, pg in zip(last_conv_layer_output, pooled_grads):
        heatmap = out @ pg[..., tf.newaxis]
        heatmap = np.maximum(heatmap, 0)     # apply ReLU to filter negative activations
        heatmaps.append(np.squeeze(heatmap))
    
    heatmaps = np.array(heatmaps) / np.max(heatmaps) # normalize to max value of all heatmaps (TODO: is this the right way? Or should each be normalized to itself)

    return heatmaps

def superimpose(img, heatmap, cam_path="cam.jpg", alpha=0.4):

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)
    img = np.uint8(255 * img)

    # Use jet colormap to colorize heatmap
    jet = matplotlib.colormaps['jet']

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Display Grad CAM
    return superimposed_img


def generate_animation(inputs, images, fps, title='', cmap='jet', alpha=0.4):
    
    # superimpose returns 8 bit image
    min_val = 0
    max_val = 255
    
    def process_map(i, alpha):
        mp = inputs[i]
        img = images[i]
        img = (img-np.min(img))/(np.max(img)-np.min(img)) # normalize image for better colors

        return superimpose(np.squeeze(img), np.squeeze(mp), alpha=alpha)

    fig = plt.figure(figsize=(10, 6))
    out = process_map(0, alpha)
    im = plt.imshow(out, vmin=min_val, vmax=max_val, cmap=cmap)
    im.axes.xaxis.set_ticks([])
    im.axes.yaxis.set_ticks([])
    fig.colorbar(im)
    plt.title(title)
    plt.tight_layout()

    plt.close()

    def init():
        out = process_map(0, alpha)
        im.set_data(out)

    def animate(i):
        out = process_map(i, alpha)
        im.set_data(out)
        return im
    
    return animation.FuncAnimation(fig, animate, init_func=init, frames=images.shape[0], interval=1/fps * 1000)

def generate_video_animation(video, fps, title=''):
    
    fig = plt.figure(figsize=(10,6))
    im = plt.imshow(video[0])
    im.axes.xaxis.set_ticks([])
    im.axes.yaxis.set_ticks([])
    fig.colorbar(im)
    plt.title(title)
    plt.tight_layout()
    
    plt.close()
    
    def init():
        im.set_data(video[0])
        
    def animate(i):
        im.set_data(video[i])
        
    return animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0], interval=1/fps * 1000)


def interpolate_video(video: np.array, target_frames: int) -> np.array:
    frames, height, width, channels = video.shape

    new_video = np.zeros([target_frames, height, width, channels], dtype='uint8')
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                x_new = np.linspace(0, frames, target_frames)
                x = np.linspace(0, frames, frames)
                y = video[:, i, j, c]
                video_inter = interp1d(x, y)
                y_new = video_inter(x_new)
                new_video[:, i, j, c] = y_new

    return new_video