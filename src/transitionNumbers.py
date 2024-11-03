from models.conditional_gan import Generator
import torch
import matplotlib.pyplot as plt
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--logs", type = float)
parser.add_argument("--checkpoint", type = float)
parser.add_argument("--x", type = float)
parser.add_argument("--y", type = float)

def main():

    channels = 1
    img_size = 28
    img_shape = (channels, img_size, img_size)
    latent_dim = 100

    args = parser.parse_args()

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    num_classes = 10
    num_interpolations = 10

    weight_path  = os.path.join(args.logs, args.checkpoint)

    generator = Generator(num_classes=num_classes).to(device)
    generator.load_state_dict(torch.load(weight_path))
    generator.eval()

    label_x = torch.tensor([args.x], device=device)
    label_y = torch.tensor([args.y], device=device)

    # Noise for the generator (same noise for all interpolations)
    noise = torch.randn(1, latent_dim, 1, 1, device=device).repeat(num_interpolations, 1, 1, 1)

    # Create interpolated labels
    gen_labels = []
    for i in range(num_interpolations):
        alpha = i / (num_interpolations - 1)  # Progress factor (0 -> 1)
        interpolated_label = (1 - alpha) * label_x + alpha * label_y
        gen_labels.append(interpolated_label)

    gen_labels = torch.stack(gen_labels).long()

    # Generate images from noise and interpolated labels
    with torch.no_grad():
        gen_images = generator(noise, gen_labels).detach().cpu().numpy()

    # Reshape generated images for visualization
    gen_images = gen_images.reshape(num_interpolations, channels, img_size, img_size)
    gen_images = gen_images.transpose(0, 2, 3, 1)  # Rearranging dimensions if needed


    # Assuming gen_images has shape (num_interpolations, channels, height, width)
    # Rearranging to (num_interpolations, height, width, channels) for visualization
    #gen_images = gen_images.transpose(0, 2, 3, 1)  # Rearranging dimensions if needed

    # Visualizing the images
    plt.figure(figsize=(num_interpolations * 2, 2))  # Adjust figure size as needed
    for i in range(num_interpolations):
        print(gen_images[i].shape)
        plt.subplot(1, num_interpolations, i + 1)  # Create a subplot for each image
        plt.imshow(gen_images[i])  # Display each generated image
        plt.axis('off')  # Hide axis
    plt.show()


if __name__ == "__main__":
    main()