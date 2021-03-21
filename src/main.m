% Clean up
clc; close all; clear;

load("monkeydata_training.mat")

[err, acc, elapsed] = testFunction_for_students_MTb(trial, "No Brain No Gain");

fprintf("Done in %.2fs: RMSE = %.2f, accuracy = %.2f. \n", elapsed, err, acc)