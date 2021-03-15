% Clean up
clc; close all; clear;

load("monkeydata_training.mat")
tic
RMSE = testFunction_for_students_MTb(trial, "No Brain No Gain");
toc