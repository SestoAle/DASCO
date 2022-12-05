import argparse
import glob
import pandas as pd
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiments, we will plot more experiments in the same plot (for each value)
    parser.add_argument("-rs", "--runs", default="sac.json")  # Name of the experiments
    parser.add_argument("-f", "--folder", default="arrays")  # Name of the Folder
    parser.add_argument("-ma", "--moving-average", default=[100, 1], type=list)  # COM value for rewards moving average

    args = parser.parse_args()

    # Get the real name of the files
    runs = args.runs
    while runs == "" or runs == " " or runs == None:
        models_name = input('Insert model name: ')

    runs = runs.replace(' ', '')
    runs = runs.replace('.json', '')
    runs = runs.split(",")

    filenames = []
    for run_name in runs:
        path = glob.glob("{}/{}.json".format(args.folder, run_name))
        for filename in path:
            with open(filename, 'r') as f:
                filenames.append(filename)

    if len(filenames) == 0:
        raise Exception("There are no files in the folder *{}* with names: {}".format(args.folder, args.runs))

    print("Here are the file names to plot: {}".format(filenames))

    # For each file name, load the stats
    stats = []
    for run in filenames:
        # Load stats file
        with open(run, 'rb') as handle:
            stats.append(json.load(handle))

    # Group stats per name
    stats_dict = dict()
    filenames = []
    for s, r in zip(stats, runs):
        name_run = r
        if name_run in stats_dict:
            stats_dict[name_run].append(s)
        else:
            stats_dict[name_run] = [s]
            filenames.append(name_run)

    # Names of the data to plot and plot titels
    data_names = ['episode_rewards']
    title_names = ['Environment Rewards']

    # Plot all data in different figures
    for i, data_name, title_name in zip(range(len(data_names)), data_names, title_names):
        plt.figure()
        for run_name, stats in stats_dict.items():
            number_of_updates = len(stats[0]['episode_rewards'])
            number_of_rm_updates = len(stats[0]['reward_model_loss'])

            all_datas = []
            for s in stats:
                data = s[data_name]
                data = pd.DataFrame({'data': data})
                data = data.ewm(com=args.moving_average[i]).mean()

                all_datas.append(data)
            all_datas = np.asarray(all_datas)

            # Compute means and stds of runs
            means = np.mean(all_datas, axis=0)
            stds = np.std(all_datas, axis=0)

            plt.title(title_name)
            if data_name == 'reward_model_loss':
                step = number_of_updates / number_of_rm_updates
                x = np.arange(0, number_of_updates, step)
                plt.plot(x, means)
            else:
                plt.plot(means)

        plt.legend(filenames)

    # Show the images
    plt.show()


