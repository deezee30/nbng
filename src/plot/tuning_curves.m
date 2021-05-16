close all

K = 8;
T = 300;
N = 100;
I = 98;

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

%figure
%imshow(fire_rate);

% Remove bottom P percentile of selective electrodes
%discard_percs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90];

%for P_idx = 1:length(discard_percs)
%P = discard_percs(P_idx);
P = 96;

[sel_idx, sel_fire_rate] = select_electrodes(fire_rate, P);

fprintf("Optimal electrodes for P = %d: %s\n", P, num2str(sel_idx));
%end

%figure
%imshow(selected_fire_rate');

% Plot individual tuning curves
SI = size(sel_fire_rate, 2); % number of selective electrodes

k_span = 1:8;
theta_span = [30, 70, 110, 150, 190, 230, 310, 350];

tcurves = figure;
hold on
for si_idx = 1:length(sel_idx)
    subplot(1, 4, si_idx);
    rate = sel_fire_rate(:, si_idx)';
    
    % extrapolate between 350 and 30 degrees
    extrap = linspace(rate(1), rate(end), 5);
    theta_span_extrap = [0, theta_span, 360];
    rate_extrap = [extrap(2), rate, extrap(2)];
    
    area(theta_span_extrap, rate_extrap, "facecolor", [0.25, 0.7, 0.6], "facealpha", 0.5, ...
                                         "edgecolor", [0.25, 0.7, 0.6])
    xticks(0:90:360)
    xlim([0, 360])
    
    title("Neuron unit #" + sel_idx(si_idx), "fontsize", 12)
end

han = axes(tcurves, "visible", "off"); 
han.Title.Visible = "on";
han.XLabel.Visible = "on";
han.YLabel.Visible = "on";
ylabel(han, "Relative spiking rate", "fontsize", 10);
xlabel(han, ["", "Stimulus"], "fontsize", 10);
title(han, ["Response of 4 most selective neuron units (96th percentile)", ""], "fontsize", 11)

xh = get(gca, "xlabel");
p = get(xh, "position");
p(2) = 0.8*p(2);

set(xh, "position", p)