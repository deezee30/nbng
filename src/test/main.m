%% Setup

% Clean up
clc; close all; clear;

% Data organisation:
% size(trial) = 100 (trials) x 8 (reaching angles)
% Each (100x8) cell in trial is a struct of 3 fields:
% - trialId: int
% - spikes: 98 (neuron) x 672 (82 ms 8 trials): either 1 (spike) or 0 (non-spike). Step: 1 ms
% - handPos: 3 (x, y, z trajectories) x 672 (82 ms x 8 trials): Relative position in cm
load("monkeydata_training.mat")

% Notes:
% 30% of units are well-isolated single-neuron units
% 70% multi-neuron units (population)

% Reaching angles for indices k
k = 1:8;
phi = [30, 70, 110, 150, 190, 230, 310, 350]/180/pi;

%% Raster plot for 1st trial, 1st angle reach, across all 98 neurons
trial_idx = 1;
angle_idx = 1;
trials = trial(trial_idx, angle_idx);
train = trials.spikes;

raster_plot(train, "Rasterplot for trial #" + trial_idx + ", angle #" + angle_idx, "Neuron #")

%% Test function
clc;

RMSE = testFunction_for_students_MTb(trial, "No Brain No Gain");