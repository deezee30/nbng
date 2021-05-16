clc
clear
close all

load("RMSE_training_size.mat");

figure
plot(fliplr(RMSES_training_size), "color", [0.25, 0.7, 0.6], "linewidth", 2)

title("RMSE Across Varying Cross-Validation Ratios")
xticklabels(["10/90", "20/80", "30/70", "40/60", "50/50", "60/40", "70/30", "80/20", "90/10"]);
xlabel("Training/Test Dataset Size");
ylabel("RMSE")