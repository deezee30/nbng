function [model_params] = positionEstimatorTraining(training_data)
    % Arguments:
    %
    % - training_data:
    %     training_data(n,k)              (n = trial id,  k = reaching angle)
    %     training_data(n,k).trialId      unique number of the trial
    %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
    %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
    %
    % ... train your model
    %
    % Return Value:
    %
    % - model_params:
    %     single structure containing all the learned parameters of your
    %     model and which can be used by the "positionEstimator" function.
    
    % Constants
    spike_dist_window = 80; % Window of spike density mixture model
    spike_dist_std    = 50; % Standard deviation of spike density mixture model
    k_num_neighbours  = 8;  % Number of k-nearest neighbours used for KNN classification
    n_bins            = 20; % Number of spike bins of a neuron
    
    n_trials = size(training_data, 1);              % Number of recorded trials
    n_trjs   = size(training_data, 2);              % Number of recorded trajectories
    n_neuron = size(training_data(1,1).spikes, 1);  % Number of neurons
    
    % plot all trials with all trajectories
    %plot_trajectories(training_data)

    % find the length of the longest trial
    L = 0;
    fprintf("\nFinding the length of longest trial... ");
    t_start = tic; % Begin ticker
    for k = 1:n_trjs
        for n = 1:n_trials    
            l = length(training_data(n,k).spikes(1,:));
            if l > L
                L = l;
            end
        end
    end
    fprintf("Done (%.2f s).\n", toc(t_start));
    
    % make all trajectories the same length
    fprintf("Making all trajectories the same length (%d pts)... ", L);
    t_start = tic; % Begin ticker
    for k = 1:n_trjs
        for n = 1:n_trials
            for j = length(training_data(n, k).spikes(1, :)) + 1:L % adjust to max length range
                training_data(n, k).handPos = [training_data(n, k).handPos ...
                                               training_data(n, k).handPos(:, end)];
                training_data(n, k).spikes  = [training_data(n, k).spikes ...
                                               training_data(n, k).spikes(:, end)];
            end
        end
    end
    fprintf("Done (%.2f s).\n", toc(t_start));
    
    edges = fix(linspace(1, L, n_bins));
    bin_size =  edges(2) - edges(1); % Bin size of spike time spans
    model_params.bin_size = bin_size;
    
    % calculate the average trajectory
    avg_trjs(n_trjs).handPos = [];
    fprintf("Calculating average trajectories... ");
    t_start = tic; % Begin ticker
    for k = 1:n_trjs
        trj = zeros(2, L);
        for n = 1:n_trials
            for j = 1:length(training_data(n, k).handPos(1, :))
                trj(:, j) = trj(:, j) + training_data(n, k).handPos(1:2, j);
            end
        end
        % Get the average trrajectory
        this_avg_trj = trj(:, :) / n_trials; % 2 x 975
        
        % Bin the average trajectory
        avg_hand_pos = zeros(2, n_bins);
        for bin = 1:n_bins-1
            traj_segment = this_avg_trj(:, edges(bin) : edges(bin+1));
            avg_hand_pos(:,bin) = mean(traj_segment');
        end
        avg_trjs(k).handPos = avg_hand_pos;
    end
    fprintf("Done (%.2f s).\n", toc(t_start));

    % plot average trajectory
    %plot_avg_trajectories(avg_trjs)
                
    % Binning spikes into n_bins bins to save processing time
    fprintf("Coverting from %d bins to %d bins... ", L, n_bins);
    t_start = tic; % Begin ticker
    for n = 1:n_trials
        for k = 1:n_trjs
            training_data(n, k).discSpikes = disc_bin(training_data(n, k).spikes(:, :), edges);
        end
    end
    fprintf("Done (%.2f s).\n", toc(t_start));
        
    % Collect spike rate (density) function for each trial
    fprintf("Collecting spike density function for each trial... ");
    t_start = tic; % Begin ticker
    x = -spike_dist_window:1:spike_dist_window;
    y = normpdf(x, 0, spike_dist_std);
    for n = 1:n_trials
        for k = 1:n_trjs
            training_data(n, k).spikeDist = zeros(n_neuron, n_bins);
            for i = 1:n_neuron
                % Mixture model of spike distribution resembling localised (time-dependent)
                % spike rate, binned at (L/n_bins) widths.
                for bin = 1:n_bins
                    if training_data(n, k).discSpikes(i, bin) > 0
                        for j = -spike_dist_window:spike_dist_window
                            if j+bin > 0 && j+bin < n_bins
                                training_data(n, k).spikeDist(i, bin+j) = training_data(n, k).discSpikes(i, bin+j) ...
                                                                        + y(j+spike_dist_window+1);
                            end
                        end
                    end
                end
            end
        end
    end
    fprintf("Done (%.2f s).\n", toc(t_start));
    
%     figure
%     plot(edges, training_data(1, 1).spikeDist(1, :))
    
    % Average spike rate (density) across all trials for each neuron
    avg_spike_rate = zeros(n_trjs, n_neuron, n_bins);
    fprintf("Calculating average spike rate across all trials for each neuron... ");
    t_start = tic; % Begin ticker
    for k = 1:n_trjs
        for i = 1:n_neuron
            for bin = 1:n_bins
                spike_sum = 0;
                for n = 1:n_trials
                    spike_sum = spike_sum + training_data(n, k).spikeDist(i, bin);
                end
                avg_spike_rate(k, i, bin) = spike_sum / n_trials;
            end
        end
    end
    fprintf("Done (%.2f s).\n", toc(t_start));
    
%     figure
%     plot(edges, squeeze(avg_spike_rate(1, 1, :)))
    
    % Linear regression between spike densities and average trajectories
    fprintf("Begin calculation of linear regression weights... ");
    t_start = tic; % Begin ticker
    pos_preds = []; % (n_bins*8) x 2
    beta = zeros(n_trjs, n_neuron, 2); % 8 x 98 x 2
    
    features = []; % 98 x (n_bins*8)
    labels = []; % 1 x (n_bins*8)
    for k = 1:n_trjs

        pos = [avg_trjs(k).handPos(1, :); avg_trjs(k).handPos(2, :)]'; % n_bins x 2
        firing_rate = [squeeze(avg_spike_rate(k, :, :))]; %  98 x n_bins
        
        features = [features , firing_rate];
        these_labels = ones(1, n_bins) .* k;
        labels = [labels , these_labels];
        
        beta_now = lsqminnorm(firing_rate', pos); % 98 x 2
                        
        beta(k, :, :) = beta_now;
                
        pos_pred = firing_rate' * beta_now; % n_bins x 2
        pos_preds = [pos_preds; pos_pred]; % (n_bins*8) x 2
        
    end
    fprintf("Done (%.2f s).\n", toc(t_start));
        
    model_params.beta = beta;
    
    % Plot prediction
%     figure(1)
%     title('Current Predictions')
%     hold on
%     for k = 1:n_trjs
%         end_point = k * n_bins;
%         
%         start_point = ( (k-1) * n_bins ) + 1;
%         pos_current = pos_preds(start_point:end_point, :);
% 
%         x_pos = pos_current(:, 1)';
%         y_pos = pos_current(:, 2)';
% 
%         plot(x_pos, y_pos, 'b')
%         hold on
%     end
%     axis square
%     hold off
%     figure
    
    features = features'; % (n_bins*8) x 98
    labels = labels'; % (n_bins*8) x 1
    Mdl = fitcknn(features, labels, 'NumNeighbors', k_num_neighbours);
    model_params.mdl = Mdl;
    
    accuracy = 0;
    for i = 1:n_bins*n_trjs
        x_in = features(i,:);
        prediction = predict(Mdl, x_in);
        if prediction == labels(i)
            accuracy = accuracy + 1;
        end
    end
    disp("Accuracy of classifier:");
    disp(accuracy/(n_bins*n_trjs));
%     x_coord = [1:1:n_bins];
%     features = features';
%     for t = 1:n_trjs
%         figure
%         for i = 1:n_neuron
%             plot(x_coord, features(i,((t-1)*n_bins) + 1 :t * n_bins), 'b')
%             hold on
%         end
%     end
    whos
end