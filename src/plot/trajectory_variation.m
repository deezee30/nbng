%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

clc
clear
close all

K = 8;
T = 570;
N = 100;

load("monkeydata_training.mat")

K_show = [4]; % Just the first direction
%K_show = 1:8; % all directions
t_span = 1:T;

x_traj = zeros(K, N, T);
y_traj = zeros(K, N, T);

% Convert struct to matrix
for k = 1:K
    for n = 1:N
        x_traj(k, n, :) = trial(n, k).handPos(1, 1:T);
        y_traj(k, n, :) = trial(n, k).handPos(2, 1:T);
    end
end

% Save mean and average standard deviations for later analysis
std_mean = zeros(K, 2);
std_min = zeros(K, 2);

figure
hold all

for ki = 1:length(K_show)
    
    k = K_show(ki);
        
    % x-coordinates
    x_mean = mean(squeeze(x_traj(k, :, :)));
    x_std = std(squeeze(x_traj(k, :, :)));
    
    std_mean(k, 1) = mean(x_std);
    std_min(k, 1) = min(x_std);

    x_upper = x_mean + x_std;
    x_lower = x_mean - x_std;

    plot(t_span, x_mean, "color", [0.7, 0.4, 1])
    patch([t_span fliplr(t_span)], [x_upper fliplr(x_lower)], [0.7, 0.4, 1], "facealpha", 0.2, "edgealpha", 0)
    
    % y-coordinates
    y_mean = mean(squeeze(y_traj(k, :, :)));
    y_std = std(squeeze(y_traj(k, :, :)));
    
    std_mean(k, 2) = mean(y_std);
    std_min(k, 2) = min(y_std);
    
    y_upper = y_mean + y_std;
    y_lower = y_mean - y_std;
    
    plot(t_span, y_mean, "color", [0.25, 0.7, 0.6])
    patch([t_span fliplr(t_span)], [y_upper fliplr(y_lower)], [0.25, 0.7, 0.6], "facealpha", 0.27, "edgealpha", 0)
end

xline(300, "k--", "linewidth", 1.5, "Alpha", 0.5);
legend(["Mean x trajectory", "", "Mean y trajectory", ""], "location", "southwest")

title(["Average trajectories for {k = 4}"])
xlabel("Reaction time (ms)")
ylabel("Position (cm)")