import torch
import numpy as np
import matplotli
def show_decoded_image(decoder, latents):
    """
    Decodes latent representations and displays them as images.

    Args:
    - decoder: The decoder model to convert latents to images.
    - latents (torch.Tensor): The latent representation tensor to decode, e.g., (1, C, H, W).
    """
    # Decode the latents
    with torch.no_grad():
        decoded_image = decoder(latents)

    # Assuming the decoded output has values in range [-1, 1]
    decoded_image = decoded_image.squeeze(0)  # Remove batch dimension if present
    image = decoded_image.detach().cpu().numpy()

    # If it's a 3-channel image, permute channels to HWC format (C, H, W -> H, W, C)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # De-normalize if needed (from [-1, 1] to [0, 1])
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)

    # Display the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()