import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import imageio
import os

plt.rcParams['figure.figsize'] = (10, 8)

# Set the path to the input folder containing images
input_folder = 'C:/Users/utsav/Desktop/Auto-PCOS Challenge/PCOSGen-train-test-split/train'

# Set the path to the output folder for saving masked images
output_folder = 'C:/Users/utsav/Desktop/Auto-PCOS Challenge/PCOSGen-train-test-split/train-masked'

def format_and_render_plot():
    '''Custom function to simplify common formatting operations for exercises. Operations include: 
    1. Turning off axis grids.
    2. Calling `plt.tight_layout` to improve subplot spacing.
    3. Calling `plt.show()` to render plot.'''
    fig = plt.gcf()
    fig.axes[0].axis('off')   
    plt.tight_layout()
    plt.show()

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok = True)

# Get a list of all image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

# Process each image
for image_file in image_files:
    # Read the image
    im = imageio.imread(os.path.join(input_folder, image_file))
    im = im.astype('float64')

    # Create a mask for bone intensity
    mask_bone = im >= 75
    im_bone = np.where(mask_bone, im, 0)

    # Convert image to suitable data type (e.g., np.uint8)
    im_bone_uint8 = im_bone.astype(np.uint8)

    # Save the masked image to the output folder
    output_file = os.path.join(output_folder, f'masked_{image_file}')
    imageio.imwrite(output_file, im_bone_uint8)