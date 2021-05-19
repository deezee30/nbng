%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

clc
clear
close all

% Plot previously-computed RMSE and TIME vs. electrodes % utilisation.
load("RMSE_electrodes.mat");

% Set up figure
figure
title("RMSE and Run Time Across Varying % of Electrodes Used")
xlabel("Significant % of Electrodes Used");
xlim([10, 100])
x = 10:10:100;

% RMSE
yyaxis left
plot(x, fliplr(RMSE_electrodes))
ylabel("RMSE")

% Run Time
yyaxis right
plot(x, x.^2)
ylabel("Run time")