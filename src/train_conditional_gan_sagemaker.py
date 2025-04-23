import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import torch
import os
import argparse

from models.conditional_gan import Generator, Discriminator, weights_init
from plot_conditional_numbers import plot_results
from torchvision.datasets.mnist import read_image_file, read_label_file

def parse_args():
    parser = argparse.ArgumentParser(description='Train Conditional GAN')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/output")
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data")
    return parser.parse_args()


def validation(batch_size, real_label, num_classes, adversarial_loss, device,
               generator, discriminator, dataloader_test):

    discriminator.eval()
    generator.eval()

    d_loss_avg = 0
    g_loss_avg = 0

    for i, (imgs, label_num) in enumerate(dataloader_test, 0):

        real_imgs = imgs.to(device).unsqueeze(1)
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
          generator, discriminator, dataloader_train, dataloader_test,
                          loss_filename, evolution_loss_filename):

    iteration = 0

    print("n_epochs", n_epochs)

    # print(epoch)
    discriminator.train()
    generator.train()

    gen_checkpoint_filename_prev = None
    det_checkpoint_filename_prev = None

    for epoch in range(n_epochs):

        print("epoch", epoch)

        d_loss_avg, g_loss_avg = validation(batch_size, real_label,
                                            num_classes, adversarial_loss, device,
               generator, discriminator, dataloader_test)

        real_batch = next(iter(dataloader_test))
        real_imgs = real_batch[0]
        real_labels = real_batch[1]

        with torch.no_grad():
            fake = generator(fixed_noise, fixed_labels).detach().cpu()

        loss_filename_full = os.path.join("logs", loss_filename)
        with open(loss_filename_full, "a") as f_write_loss:
                loss_str = f"{epoch}, {d_loss_avg}, {g_loss_avg}\n"
                f_write_loss.write(loss_str)

        discriminator.train()
        generator.train()

        for i, (imgs, label_num) in enumerate(dataloader_train, 0):

            iteration += 1

            #print(f"iteration {i} / {len(dataloader_train)}")

            real_imgs = imgs.to(device).unsqueeze(1)
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

            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)

            fake_imgs = generator(noise, fake_labels)
            label.fill_(fake_label)

            fake_output = discriminator(fake_imgs.detach(), fake_labels).view(-1)
            fake_loss = adversarial_loss(fake_output, label)

            fake_loss.backward()

            if (iteration%1 ==0):

                optimizer_D.step()

            # ------------
            # Train Generator
            # ------------
            optimizer_G.zero_grad()
            label.fill_(real_label)
            fake_output = discriminator(fake_imgs, fake_labels).view(-1)

            g_loss = adversarial_loss(fake_output, label)
            g_loss.backward()

            optimizer_G.step()

            loss_filename_full = os.path.join("logs", evolution_loss_filename)
            with open(loss_filename_full, "a") as f_write_loss:
                loss_str = f"{iteration}, {fake_loss.item()}, {g_loss.item()}\n"
                f_write_loss.write(loss_str)

        # ------------
        # Writing checkpoint
        # ------------
        if (epoch % 5 == 0):

            fname = "logs/results_conditional_gan/dataset_im_epoch" + str(epoch) + ".png"
            plot_results(fname, real_imgs, real_labels, fake, fixed_labels)

            gen_checkpoint_filename = f"logs/cond_generator.epoch_{epoch}.pth"
            det_checkpoint_filename = f"logs/cond_discriminator.epoch_{epoch}.pth"

            torch.save(generator.state_dict(), gen_checkpoint_filename)

            if gen_checkpoint_filename_prev is not None:
                os.remove(gen_checkpoint_filename_prev)
            gen_checkpoint_filename_prev = gen_checkpoint_filename

            torch.save(discriminator.state_dict(), det_checkpoint_filename)

            if det_checkpoint_filename_prev is not None:
                os.remove(det_checkpoint_filename_prev)
            det_checkpoint_filename_prev = det_checkpoint_filename

    return generator, discriminator

if __name__ == '__main__':

    args = parse_args()

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

    n_epochs = args.n_epochs

    real_label = 1
    fake_label = 0
    num_samples_per_class = 5000

    loss_filename = "loss_conditional_gan.txt"
    evolution_loss_filename = "evolution_loss_conditional_gan.txt"

    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    if torch.cuda.is_available() and ngpu > 0:
        print("CUDA!")
        #Tensor = torch.cuda.FloatTensor
    else:
        print("NO CUDA!")
        #Tensor = torch.FloatTensor

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
    #transform = transforms.Compose([
    #    transforms.Resize(image_size),
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.5,), (0.5,))
    #])

    #train_dataset = datasets.MNIST(root='data', train=True,
    #                               download=True, transform=transform)

    #test_dataset = datasets.MNIST(root='data', train=False,
    #                              download=True, transform=transform)    

    #dataloader_train = DataLoader( train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    #dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    #real_batch = next(iter(dataloader_test))
    #real_imgs = real_batch[0]
    #real_labels = real_batch[1]


    #print(real_imgs.shape)
    #print(real_labels.shape)

    ####################################################

     # ------------
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Paths to the saved files
    train_images_path = 'data/MNIST/raw/train-images-idx3-ubyte'
    train_labels_path = 'data/MNIST/raw/train-labels-idx1-ubyte'
    test_images_path = 'data/MNIST/raw/t10k-images-idx3-ubyte'
    test_labels_path = 'data/MNIST/raw/t10k-labels-idx1-ubyte'

    # Load the data from the files
    train_images = read_image_file(train_images_path).float()
    train_labels = read_label_file(train_labels_path).long()
    test_images = read_image_file(test_images_path).float()
    test_labels = read_label_file(test_labels_path).long()

    # Apply transformations
    train_images = transform(train_images)
    test_images = transform(test_images)

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    # Create dataloaders
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    ####################################################

    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=lr, momentum=0.9)
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=lr, momentum=0.9)

    #optimizer_G = torch.optim.Adam(generator.parameters(), lr = lr, betas = (b1, b2))
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = lr, betas = (b1, b2))

    fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
    fixed_labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # --------
    # plot untrained generator output
    # --------
    real_batch = next(iter(dataloader_test))
    real_imgs = real_batch[0].unsqueeze(1)
    real_labels = real_batch[1]

    with torch.no_grad():
        fake = generator(fixed_noise, fixed_labels).detach().cpu()

    fname = "logs/results_conditional_gan/dataset_im_start.png"

    #print(real_imgs.shape)
    #print(real_labels.shape)

    plot_results(fname, real_imgs, real_labels, fake, fixed_labels)

    # --------
    # train GAN
    # --------

    generator,  discriminator = \
            train_conditional_GAN(n_epochs, batch_size, real_label, num_classes,
            adversarial_loss, device, optimizer_G, optimizer_D,
            generator, discriminator,  dataloader_train, dataloader_test,
                                  loss_filename, evolution_loss_filename)
    # --------
    # Plot trained generator output
    # --------
    real_batch = next(iter(dataloader_test))
    real_imgs = real_batch[0]
    real_labels = real_batch[1]

    with torch.no_grad():
        fake = generator(fixed_noise, fixed_labels).detach().cpu()

    fname = "logs/results_conditional_gan/dataset_im_fin.png"
    plot_results(fname, real_imgs, real_labels, fake, fixed_labels)