import matplotlib.pyplot as plt
import itertools

def plot_results(fname, real_imgs, real_labels, fake, fake_labels):
    #########################################
    real_imgs = real_imgs.cpu()
    fake = fake.cpu()
    real_labels = real_labels.cpu()
    fake_labels = fake_labels.cpu()

    ncols = 6
    fig, axes = plt.subplots(ncols=ncols, figsize=(8, 8))

    for ncol in range(ncols // 2):

        axes[ncol].imshow(real_imgs.permute(0, 2, 3, 1)[ncol], cmap='gray')
        axes[ncol].axis('off')
        axes[ncol].set_title('real, ' + str(real_labels[ncol].item()))

        axes[ncol + ncols // 2].imshow(fake.permute(0, 2, 3, 1)[ncol + ncols // 2], cmap='gray')
        axes[ncol + ncols // 2].axis('off')
        axes[ncol + ncols // 2].set_title('fake, ' + str(fake_labels[ncol + ncols // 2].item()))

    plt.savefig(fname)
    plt.close()

    #plt.show()