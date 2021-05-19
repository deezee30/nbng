%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

% Clean up
clc; close all; clear;

load("monkeydata_training.mat")

[N, K] = size(trial);
[I, ~] = size(trial(1, 1).spikes);

T = 300; % movement planning

tuning = zeros(K);
spikes = zeros(K, N, I, T);

% Convert struct to 4D matrix
for k = 1:K
    for n = 1:N
        spikes(k, n, :, :) = trial(n, k).spikes(:, 1:T);
    end
end

spike_trial_avg = squeeze(mean(spikes, 2)); % average out across all trials
spike_event_avg = squeeze(mean(spike_trial_avg, 3)); % average out across all time span

% Normalise based on max firing rate
max_rate = max(spike_event_avg, [], "all");
fire_rate = spike_event_avg' / max_rate; % normalise


% For plotting the entire tuning matrix. uncomment
% figure
% FR = imagesc(fire_rate');
% ax = gca;
% ax.FontSize = 10;
% xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98])
% yticks([30, 350])
% yticklabels([30, 350])
% xlabel("Electrode", "fontsize", 10.5)
% ylabel("Direction", "fontsize", 10.5)
% axis image
% set(gca,'YDir','normal')
% 
% %raw = [0.67, 0.22, 1;
% %       0.21, 0.89, 0.73];
% raw = [255, 255, 255;
%        13,  186, 169] / 255;
% cmap = interp1([100, 0], raw, linspace(100, 0, 128), 'pchip');
% colormap(gca, cmap);
% cbar = colorbar(gca, "southoutside");
% cbar.Label.String = 'Relative Firing Rate';
% 
% if true
%     return
% end

% Remove bottom P percentile of selective electrodes
%discard_percs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90];

%for P_idx = 1:length(discard_percs)
%P = discard_percs(P_idx);
P = 0;

[sel_idx, sel_fire_rate] = select_electrodes(fire_rate, P);

fprintf("%i optimal electrodes for P = %d: %s\n", length(sel_idx), P, num2str(sel_idx));
%end

%figure
%imshow(selected_fire_rate');

% Plot individual tuning curves
SI = size(sel_fire_rate, 2); % number of selective electrodes

k_span = 1:8;
theta_span = [30, 70, 110, 150, 190, 230, 310, 350];

tcurves = figure;
M = 7; % rows
N = 14; % cols
tiledlayout(M, N, 'TileSpacing', 'compact'); % all
for si_idx = 1:length(sel_idx)
    nexttile
%    subplot(7, 14, si_idx)
    rate = sel_fire_rate(:, si_idx)';
    
    % extrapolate between 350 and 30 degrees
    extrap = linspace(rate(1), rate(end), 5);
    theta_span_extrap = [0, theta_span, 360];
    rate_extrap = [extrap(2), rate, extrap(2)];
    
    area(theta_span_extrap, rate_extrap, "facecolor", [0.25, 0.7, 0.6], "facealpha", 0.5, ...
                                         "edgecolor", [0.25, 0.7, 0.6], "linewidth", 1.5)

    xticks(0:180:360)
    xlim([0, 360])
    ylim([0, 1])
    
    % When plotting all tuning curves:
    % left column
    if mod(si_idx, N) - 1 ~= 0
        ylabel([])
        yticklabels([])
    end
    % bottom row
    if si_idx <= N * (M-1)
        xticks([])
        xlabel([])
        xticklabels([])
    end
    
    ax = gca;
    ax.FontSize = 13;
    
    title(sel_idx(si_idx), "fontsize", 11)
end

han = axes(tcurves, "visible", "off", "position", [0.06 0.08 0.9 0.9]); 
han.Title.Visible = "on";
han.XLabel.Visible = "on";
han.YLabel.Visible = "on";
ylabel(han, "Relative spiking rate", "fontsize", 10);
xlabel(han, "Stimulus", "fontsize", 10);
%title(han, ["Response of 4 most selective neuron units (96th percentile)", ""], "fontsize", 11)

xh = get(gca, "xlabel");
p = get(xh, "position");
p(2) = 0.8*p(2);
set(xh, "position", p)