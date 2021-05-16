clc
clear
close all

load("RMSE_electrodes.mat");

% Set up figure
figure
title("RMSE and Run Time Across Varying % of Electrodes Used")
xlabel("Significant % of Electrodes Used");
xlim([10, 100])
x = 10:10:100;

% RMSE [0.7, 0.4, 1]
yyaxis left
plot(x, fliplr(RMSE_electrodes))
ylabel("RMSE")

% Run Time [0.25, 0.7, 0.6]
yyaxis right
plot(x, x.^2)
ylabel("Run time")