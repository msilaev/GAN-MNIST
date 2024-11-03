import numpy as np
import torch
from torch.utils.data import DataLoader


def generate_train_fake(device, generator, num_classes, latent_dim,  num_samples_per_class):
    # ------------
    # Generate test data set2
    # ------------
    generator.eval()
    generated_samples = []
    generated_labels = []
    # num_samples_per_class = 1000

    for label in range(num_classes):

        noise = torch.randn(num_samples_per_class,
                            latent_dim, 1, 1,
                            device=device)

        gen_labels = torch.full((num_samples_per_class,),
                                label, dtype=torch.long,
                                device=device)

        gen_images = generator(noise, gen_labels).detach().cpu().numpy()

        gen_images = gen_images.reshape(num_samples_per_class, -1)

        generated_samples.append(gen_images)

        generated_labels.extend([label] * num_samples_per_class)

    generated_samples = np.vstack(generated_samples)
    generated_labels = np.array(generated_labels)

    return generated_labels, generated_samples

def generate_test_train_real(dataset):
    # ------------
    # Form training or test set from real images
    # ------------

    batch_size = 1  # suggested default, size of the batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    real_images = []
    real_labels = []

    for i, (imgs, label_num) in enumerate(dataloader, 0):
        real_images.append(imgs[0])
        real_labels.append(label_num[0])

    real_images = np.array(real_images)
    real_labels = np.array(real_labels)
    real_train_images = real_images.reshape(real_images.shape[0], -1)

    return real_labels, real_train_images