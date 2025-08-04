# image_utils.py

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def tensor_to_image(tensor: torch.Tensor, target_size=(1280, 1280)) -> Image.Image:
    """
    Converts a latent vector tensor into a visually abstract PNG image.
    The actual tensor data is losslessly encoded into the image's pixel values.
    """
    # 1. Convert tensor to a float32 NumPy array on the CPU
    tensor_np = tensor.squeeze(0).cpu().numpy().astype(np.float32)
    
    # 2. Convert tensor data to a byte stream for preservation
    tensor_bytes = tensor_np.tobytes()
    
    # 3. Create a blank NumPy array for the 1280x1280 RGBA image
    image_array = np.zeros((target_size[1], target_size[0], 4), dtype=np.uint8)
    
    # 4. Write the tensor bytes directly into the image array
    byte_array = np.frombuffer(tensor_bytes, dtype=np.uint8)
    image_array.flat[:len(byte_array)] = byte_array
    
    # --- 5. Create the visual abstraction (Corrected Section) ---
    # To visualize the (128, 768) tensor, first treat it as a grayscale image
    # and resize it to a manageable square (e.g., 256x256)
    vis_img = Image.fromarray(tensor_np)
    vis_img_resized = vis_img.resize((256, 256), Image.Resampling.BICUBIC)
    
    # Convert back to a NumPy array for color mapping
    vis_array = np.array(vis_img_resized)
    
    # Normalize the array to the 0-1 range for the colormap
    normalized_array = (vis_array - vis_array.min()) / (vis_array.max() - vis_array.min())
    
    # Apply a colormap to convert the grayscale values to RGB colors
    colored_array = (plt.get_cmap('viridis')(normalized_array)[:, :, :3] * 255).astype(np.uint8)
    
    # Create the final visual PIL Image from the colored array
    final_visual_img = Image.fromarray(colored_array)
    
    # Place the generated abstract image in the center of the large canvas
    vis_w, vis_h = final_visual_img.size
    start_x = (target_size[0] - vis_w) // 2
    start_y = (target_size[1] - vis_h) // 2
    
    # This assignment now works because both sides have 3 channels (RGB)
    image_array[start_y:start_y+vis_h, start_x:start_x+vis_w, :3] = np.array(final_visual_img)
    # Set the alpha channel for the visual part to be fully opaque
    image_array[start_y:start_y+vis_h, start_x:start_x+vis_w, 3] = 255
    
    # Convert the final NumPy array to a PIL Image object
    return Image.fromarray(image_array, 'RGBA')

# The image_to_tensor function remains the same as it correctly reads the byte data
def image_to_tensor(image: Image.Image, original_shape=(128, 768)) -> torch.Tensor:
    image_array = np.array(image)
    image_bytes = image_array.tobytes()
    required_bytes = original_shape[0] * original_shape[1] * 4
    tensor_bytes = image_bytes[:required_bytes]
    tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32).reshape(original_shape)
    return torch.from_numpy(tensor_np).unsqueeze(0)