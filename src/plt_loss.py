import matplotlib.pyplot as plt

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--fname", type = str)
parser.add_argument("--type_gan", type = str)
parser.add_argument("--logs", type = str)

def main():

    args = parser.parse_args()
    file_name_full =  args.logs + "/" + args.fname + ".txt"
    epoch_arr = []
    d_loss_arr = []
    g_loss_arr = []

    with open(file_name_full, "r") as f:
        for line in f:
            epoch, d_loss, g_loss = line.split(",")
            epoch_arr.append(float(epoch))
            d_loss_arr.append(float(d_loss))
            g_loss_arr.append(float(g_loss))

    d_loss_arr = np.array(d_loss_arr)
    g_loss_arr = np.array(g_loss_arr)
    epoch_arr = np.array(epoch_arr)

    #print(d_loss_arr)

    plt.figure()

    plt.figure(figsize=(5, 5))  # Adjust width and height as needed

    # Set font sizes globally
    plt.rcParams.update({'font.size': 20})  # General font size
    plt.rcParams.update({'axes.titlesize': 20})  # Title font size
    plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

    plt.plot(epoch_arr, d_loss_arr, color = "orange", marker = "+", label = "Discr")
    plt.plot(epoch_arr, g_loss_arr, color="blue", marker="o", label = "Gen")
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Adv Loss")
    plt.ylim([0,5])
    plt.xlim([0, 100])
    plt.tight_layout()


    if args.type_gan == "conditional":
        figname = os.path.join(args.logs, "results_conditional_gan", args.fname + ".png")
           # "logs/results_conditional_gan/" + args.fname + ".png"

    elif args.type_gan == "unconditional":
        figname = os.path.join(args.logs, "results_unconditional_gan", args.fname + ".png")
        #figname = "logs/results_unconditional_gan/" + args.fname + ".png"

    plt.savefig(figname, format = "png")
    plt.close()

if __name__ == "__main__":
    main()
