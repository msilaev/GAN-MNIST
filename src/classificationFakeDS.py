import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torchvision import datasets

import torch
import matplotlib.pyplot as plt
import os
import argparse

from models.conditional_gan import Generator, Discriminator
from models.classifier import Classifier
#from plot_conditional_numbers import plot_results
from models.dataset_factory import generate_train_fake, generate_test_train_real

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type = str)
parser.add_argument("--logs", type = str)

if __name__ == '__main__':

    args = parser.parse_args()

    channels = 1
    img_size = 28
    img_shape = (channels, img_size, img_size)
    latent_dim = 100

    num_classes = 10
    image_size = 28
    batch_size = 64

    batch_size = 64
    b1 = 0.5
    b2 = 0.999
    lr = 0.0002

    real_label = 1
    fake_label = 0
    num_samples_per_class = 5000

    path_generator = os.path.join(args.logs, args.checkpoint)

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if torch.cuda.is_available() and ngpu > 0:
        print("CUDA!")
        Tensor = torch.cuda.FloatTensor
    else:
        print("NO CUDA!")
        Tensor = torch.FloatTensor

    # --------
    # Define loss function, initialize generator and discriminator
    # --------
    adversarial_loss = torch.nn.BCELoss()

    generator = Generator(num_classes = num_classes).to(device)

    # --------
    # Generate fake dataset using untrained generator
    # --------
    generated_labels, generated_samples = \
        generate_train_fake(device, generator, num_classes,
                            latent_dim, num_samples_per_class)

    #generate_train_fake(generator, num_classes, num_samples_per_class)

    # ------------
    # Download MNIST train and test dataset from PyTorch
    # ------------

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)

    test_dataset = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

    real_train_labels, real_train_images = generate_test_train_real(train_dataset)

    real_test_labels, real_test_images = generate_test_train_real(test_dataset)

    # --------
    # Calculate accuracy of KNN classifier with untrained generator and real dataset
    # --------

    accuracy_generated = Classifier(real_test_labels, real_test_images,
                      generated_labels, generated_samples, num_classes)

    accuracy_real = Classifier(real_test_labels, real_test_images,
                   real_train_labels, real_train_images, num_classes)

    print(f"Error rate using real dataset = {100*(1- accuracy_real):.2f}%")
    print(f"Error rate using untrained generator dataset = {100*(1- accuracy_generated):.2f}%")

    # ------------
    # Alternatively we can download dataset from PyTorch
    # ------------

    #transform = transforms.Compose([
    #    transforms.Resize(image_size),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.5,), (0.5,))
    #])

    #mnist_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    #dataloader = DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

    dataloader = DataLoader( train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas = (b1, b2))

    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fixed_labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # --------
    # plot untrained generator output
    # --------
    real_batch = next(iter(dataloader))
    real_imgs = real_batch[0]
    real_labels = real_batch[1]

    with torch.no_grad():
        fake = generator(fixed_noise, fixed_labels).detach().cpu()

    #fname = "../results_conditional_gan/dataset_im_start.png"
    #plot_results(fname, real_imgs, real_labels, fake, fixed_labels)
    #loss_filename = os.path.join("logs","loss.txt")

    # Load the generator model
    generator.load_state_dict(torch.load(path_generator))

    # --------
    # Ganerate fake dataset using trained generator
    # --------
    generated_labels, generated_samples = \
        generate_train_fake(device, generator, num_classes, latent_dim, num_samples_per_class)

    # --------
    # Calculate accuracy of KNN classifier with trained generator
    # --------
    accuracy_generated \
        = Classifier(real_test_labels, real_test_images, generated_labels, generated_samples, num_classes)

    print(f"Error rate using trained generator dataset = {100*(1- accuracy_generated):.2f}%")