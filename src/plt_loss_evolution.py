import matplotlib.pyplot as plt

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type = str)
parser.add_argument("--type_gan", type = str)
parser.add_argument("--logs", type = str)


# evolution_loss_conditional_gan.txt"

def main():

    args = parser.parse_args()

    file_name_full = args.logs + "/" + args.fname + ".txt"
    epoch_arr = []
    d_loss_arr = []
    g_loss_arr = []

    # with is a context manager
    # __enter__ and __exit__
    # when use open() __enter__ onpens file, __exit__ closes

    with open(file_name_full, "r") as f:
        for line in f:
            epoch, d_loss, g_loss = line.split(",")
            epoch_arr.append(float(epoch))
            d_loss_arr.append(float(d_loss))
            g_loss_arr.append(float(g_loss))

    d_loss_arr = np.array(d_loss_arr)
    g_loss_arr = np.array(g_loss_arr)

    window_size = 50  # Set the window size for moving average

    # Calculate moving average using convolution
    d_loss_arr = np.convolve(d_loss_arr, np.ones(window_size) / window_size, mode='valid')
    g_loss_arr = np.convolve(g_loss_arr, np.ones(window_size) / window_size, mode='valid')

    epoch_arr = np.arange(len(g_loss_arr))

    plt.figure()

    plt.figure(figsize=(10, 4))  # Adjust width and height as needed

    # Set font sizes globally
    plt.rcParams.update({'font.size': 14})  # General font size
    plt.rcParams.update({'axes.titlesize': 18})  # Title font size
    plt.rcParams.update({'axes.labelsize': 16})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 14})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 14})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 14})  # Y tick label font size

    plt.plot(epoch_arr, d_loss_arr, color = "orange",  label = "discriminator")
    plt.plot(epoch_arr, g_loss_arr, color="blue",  label = "generator")
    plt.legend()
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("<Adv Loss>_50")
    plt.ylim([0,5])
    plt.tight_layout()

    if args.type_gan == "conditional":
        figname = os.path.join(args.logs, "results_conditional_gan", args.fname + ".png")
        #figname = "logs/results_conditional_gan/" + args.fname + ".png"

    elif args.type_gan == "unconditional":
        figname = os.path.join(args.logs, "results_unconditional_gan", args.fname + ".png")
        #figname = "logs/results_unconditional_gan/" + args.fname + ".png"

    plt.savefig(figname, format = "png")
    plt.close()

if __name__ == "__main__":
    main()
