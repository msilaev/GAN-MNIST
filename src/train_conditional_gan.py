import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import datasets

import torch
import matplotlib.pyplot as plt
import os

from models.conditional_gan import Generator, Discriminator, weights_init
from dataset_batch import DatasetMNIST
from classifier import Classifier
from plot_results import plot_results
from dataset_factory import generate_train_fake, generate_test_train_real

def validation(batch_size, real_label, num_classes, adversarial_loss, device,
               generator, discriminator, dataloader_test):

    discriminator.eval()
    generator.eval()

    #d_loss_arr.append(d_loss.item())
    #g_loss_arr.append(g_loss.item())

    d_loss_avg = 0
    g_loss_avg = 0

    for i, (imgs, label_num) in enumerate(dataloader_test, 0):

        real_imgs = imgs.to(device)
        labels = label_num.to(device)

        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

        valid_labels = labels
        fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)

        real_output = discriminator(real_imgs, valid_labels).view(-1)

        real_loss = adversarial_loss(real_output, label)

        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

        fake_imgs = generator(noise, fake_labels)
        label.fill_(fake_label)

        fake_output = discriminator(fake_imgs.detach(), fake_labels).view(-1)
        fake_loss = adversarial_loss(fake_output, label)

        d_loss = (real_loss + fake_loss) / 2

        label.fill_(real_label)
        fake_output = discriminator(fake_imgs, fake_labels).view(-1)

        g_loss = adversarial_loss(fake_output, label)

        d_loss_avg += d_loss.item() /len(dataloader_test)
        g_loss_avg += g_loss.item() / len(dataloader_test)

    return d_loss_avg, g_loss_avg

def train_conditional_GAN(n_epochs, batch_size, real_label, num_classes,
          adversarial_loss, device, optimizer_G, optimizer_D,
          generator, discriminator, dataloader_train, dataloader_test, loss_filename):

    #n_epochs = 1
    iteration = 0

    print("n_epochs", n_epochs)

    # print(epoch)
    discriminator.train()
    generator.train()

    for epoch in range(n_epochs):

        print("epoch", epoch)

        d_loss_avg, g_loss_avg = validation(batch_size, real_label, num_classes, adversarial_loss, device,
               generator, discriminator, dataloader_test)

        ############################3333

        loss_filename_full = os.path.join("logs", loss_filename)


        with open(loss_filename_full, "a") as f_write_loss:

                loss_str = f"{epoch}, {d_loss_avg.item()}, {g_loss_avg.item()}\n"

                f_write_loss.write(loss_str)

        #print(
        #    "[Epoch: %d/%d] [Batch: %d/%d] [D loss: %f] [G loss: %f]"
        #    % (epoch + 1, n_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item())
        #)

        #################################

        for i, (imgs, label_num) in enumerate(dataloader_train, 0):

            iteration += 1

            print(f"iteration {i} / {len(dataloader_train)}")

            real_imgs = imgs.to(device)
            labels = label_num.to(device)

            # ------------
            # Train Discriminator
            # ------------
            optimizer_D.zero_grad()

            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

            valid_labels = labels
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)

            real_output = discriminator(real_imgs, valid_labels).view(-1)

            real_loss = adversarial_loss(real_output, label)
            real_loss.backward()

            D_real = real_output.mean().item()

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

            fake_imgs = generator(noise, fake_labels)
            label.fill_(fake_label)

            fake_output = discriminator(fake_imgs.detach(), fake_labels).view(-1)
            fake_loss = adversarial_loss(fake_output, label)

            fake_loss.backward()
            D_fake = fake_output.mean().item()

            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.step()

            # ------------
            # Train Generator
            # ------------
            optimizer_G.zero_grad()
            label.fill_(real_label)
            fake_output = discriminator(fake_imgs, fake_labels).view(-1)

            g_loss = adversarial_loss(fake_output, label)
            g_loss.backward()
            D_G_fake = fake_output.mean().item()
            optimizer_G.step()

            if ((iteration + 1) % 500) == 0:

                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, n_epochs, i, len(dataloader),
                         d_loss.item(), g_loss.item(), D_real, D_fake, D_G_fake))


    return generator, discriminator


if __name__ == '__main__':

    channels = 1
    img_size = 28
    img_shape = (channels, img_size, img_size)
    latent_dim = 100

    num_classes = 10
    image_size = 28
    batch_size = 64

    batch_size = 64
    b1 = 0.9
    b2 = 0.999
    lr = 0.0002

    real_label = 1
    fake_label = 0
    num_samples_per_class = 5000

    path_generator = os.path.join("models", 'generator.pth')
    path_discriminator = os.path.join("models", 'discriminator.pth')

    loss_filename = "loss_conditional_gan.txt"

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
    generator.apply(weights_init)

    discriminator = Discriminator(num_classes = num_classes,
                                  image_size = image_size).to(device)
    discriminator.apply(weights_init)

    #train_dataset = DatasetMNIST(file_path = '../data/train.csv',
    #                       transform = transforms.Compose( [transforms.ToTensor(),
    #                        transforms.Normalize([0.5], [0.5])]))

    #test_dataset = DatasetMNIST(file_path = '../data/test.csv',
    #                       transform = transforms.Compose( [transforms.ToTensor(),
    #                        transforms.Normalize([0.5], [0.5])]))


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

    dataloader_train = DataLoader( train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas = (b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (b1, b2))

    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fixed_labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # --------
    # plot untrained generator output
    # --------
    real_batch = next(iter(dataloader_test))
    real_imgs = real_batch[0]
    real_labels = real_batch[1]

    with torch.no_grad():
        fake = generator(fixed_noise, fixed_labels).detach().cpu()

    fname = "../results_conditional_gan/dataset_im_start.png"

    plot_results(fname, real_imgs, real_labels, fake, fixed_labels)

    # --------
    # train GAN
    # --------
    n_epochs = 100

    generator,  discriminator = \
            train_conditional_GAN(n_epochs, batch_size, real_label, num_classes,
            adversarial_loss, device, optimizer_G, optimizer_D,
            generator, discriminator,  dataloader_train, dataloader_test, loss_filename)

    torch.save(generator.state_dict(), path_generator)
    torch.save(discriminator.state_dict(), path_discriminator)

    #f_write_loss = open(loss_filename, "w")

    #for ind, i  in enumerate(d_loss_arr):
    #        loss_str = f"{ind}, {d_loss_arr[ind]}, {g_loss_arr[ind]}\n"
    #        f_write_loss.write(loss_str)
    #f_write_loss.close()

    # --------
    # plot learning curves
    # --------
    #d_loss_arr = np.array(d_loss_arr)
    #g_loss_arr = np.array(g_loss_arr)
    #g_loss_arr_1 = - np.log(1 - np.exp(-g_loss_arr))

    #plt.figure(figsize=(10, 5))
    #plt.title("Generator and Discriminator Loss During Training")
    #plt.plot(g_loss_arr_1, label="G")
    #plt.plot(d_loss_arr, label="D")
    #plt.xlabel("iterations")
    #plt.ylabel("Loss")
    #plt.grid(True)  # Add grid lines
    #plt.legend()

    #plt.savefig("../results_conditional_gan/loss.png")
    #plt.show()

    # --------
    # Plot trained generator output
    # --------
    real_batch = next(iter(dataloader_test))
    real_imgs = real_batch[0]
    real_labels = real_batch[1]

    with torch.no_grad():
        fake = generator(fixed_noise, fixed_labels).detach().cpu()

    fname = "../results_conditional_gan/dataset_im_fin.png"
    plot_results(fname, real_imgs, real_labels, fake, fixed_labels)

    # --------
    # Ganerate fake dataset using trained generator
    # --------

    #generated_labels, generated_samples = \
    #    generate_train_fake(device, generator, num_classes, latent_dim, num_samples_per_class)

    # --------
    # Calculate accuracy of KNN classifier with trained generator
    # --------

    #accuracy_generated \
    #    = Classifier(real_test_labels, real_test_images, generated_labels, generated_samples, num_classes)

    #print(f"Error rate using trained generator dataset = {100*(1- accuracy_generated):.2f}%")

    #arithmetics(generator, latent_dim, device, 5, 4)





