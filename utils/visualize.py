
# Assuming you have defined train_loader and it's ready to use.

import matplotlib.pyplot as plt

def visualize_batch(data_loader, num_images=8, norm_range='-1_to_1'):
    """
    Visualizes a specified number of images from the first batch of the DataLoader.

    Args:
        data_loader (torch.utils.data.DataLoader): The DataLoader to sample from.
        num_images (int): The number of images to display.
        norm_range (str): The range the data was normalized to. 
                        Use '-1_to_1' for [-1, 1] (standard DDPM practice) or '0_to_1'.
    """
    print("Dataset Examination...")
    # 1. Get a batch
    batch_data = next(iter(data_loader))
    
    # Handle tuple/list output (e.g., (data, labels))
    if isinstance(batch_data, (list, tuple)):
        images = batch_data[0].cpu() # Assuming data is the first element
    else:
        images = batch_data.cpu()

    # Select the first 'num_images'
    images = images[:num_images]

    # Set up the plot
    fig, axes = plt.subplots(1, num_images, figsize=(1.5 * num_images, 2))

    for i in range(num_images):
        img = images[i]

        # 2. Denormalize and Permute
        if norm_range == '-1_to_1':
            # DDPM standard: scale from [-1, 1] to [0, 1]
            img = (img + 1) / 2
        elif norm_range == '0_to_1':
            # Scale from [0, 1] (if you used this range)
            pass
        else:
            raise ValueError("Invalid norm_range. Use '-1_to_1' or '0_to_1'.")

        # PyTorch format is (C, H, W). Matplotlib needs (H, W, C) for color images.
        img = img.permute(1, 2, 0) 
        
        # Convert to NumPy
        img_np = img.numpy()

        # 3. Plot the image
        ax = axes[i]
        ax.imshow(img_np.clip(0, 1)) # .clip(0, 1) ensures valid color values
        ax.axis('off')

    plt.suptitle("Sampled Images from DataLoader (Denormalized)", y=1.05, fontsize=14)
    plt.tight_layout()
    plt.savefig('test_batch.png')

# --- Example Usage (Assuming you have 'train_loader' defined) ---
# Check your normalization in your data loading/transform code. 
# DDPMs typically normalize CIFAR-10 to [-1, 1].
# If you used transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
# then use norm_range='-1_to_1'.
