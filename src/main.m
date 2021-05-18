%% Setup

% Clean up
clc; close all; clear;

% Remove model paths
rmpath(genpath("src/models"))

% Add necessary paths
addpath("data")
addpath("src")
addpath("src/plot")
addpath("src/util")
addpath("src/kernel")

load("monkeydata_training.mat")

%% Settings

model       = "svm_4"; % model name (corresponding to folder)
seeds       = [2013]; % seeds for random permutations
data_splits = [.8]; % cross validation training/testing ratios: 0.8 -> 80/20

%% Decoding

t0 = tic;

[RMSE, runtime] = decoder(trial, model, true, seeds, data_splits);

dt = toc(t0);
N = length(RMSE);

fprintf("Finished %i runs in %.2fs.\n", N, dt)