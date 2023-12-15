import os
import torchvision.transforms as T
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, colors

# Define the directory where the images are located
image_dir = 'E:/Coding_path/DiffuseVAE/scripts/reconstruction_samples/original/converted_TI_20000/'

# Initialize an empty list to store the processed images
processed_images = []

# Define the spacing between images
spacing = 0  # Adjust this value to control the spacing

# Define the transformation
transform = T.Compose([
    T.Resize((128, 128)),
    T.Grayscale(num_output_channels=1),
    T.ToTensor(),
    T.Lambda(lambda x: (x > 0.5).float()),
])

# Process and collect the images
for i in range(16):
    filename = f'crop{i}.jpg'
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path).convert('L')
    img = transform(image)
    img_binary = torch.where(img >= 0.5, torch.tensor(1.0), torch.tensor(0.0))
    img_data = img_binary.detach().cpu().numpy()
    img_data = np.squeeze(img_data, axis=0)
    processed_images.append(img_data)

# Calculate the size of the combined image
image_size = processed_images[0].shape[0]
combined_size = (image_size + spacing) * 4 - spacing

# Create the combined image with spacing
combined_image = np.zeros((combined_size, combined_size), dtype=np.uint8)

for i in range(4):
    for j in range(4):
        x_start = i * (image_size + spacing)
        x_end = (i + 1) * (image_size + spacing) - spacing
        y_start = j * (image_size + spacing)
        y_end = (j + 1) * (image_size + spacing) - spacing

        combined_image[x_start:x_end, y_start:y_end] = (
            processed_images[i * 4 + j] * 255
        ).astype(np.uint8)
# Create a custom colormap
cmap = colors.ListedColormap(['black', 'white'])

# Display the combined image
plt.imshow(combined_image, cmap=cmap)
plt.axis('off')  # Turn off axis labels

# Save the combined image with DPI set to 500
save_path = 'E:/Coding_path/DiffuseVAE/scripts/reconstruction_samples/original/converted_TI_20000/combined_image.jpg'
plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0)

# Show the image (optional)
plt.show()

