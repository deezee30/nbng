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
    k_num_neighbours  = 7;  % Number of k-nearest neighbours used for KNN classification
    
    n_trials = size(training_data, 1);              % Number of recorded trials
    n_trjs   = size(training_data, 2);              % Number of recorded trajectories
    n_neuron = size(training_data(1,1).spikes, 1);  % Number of neurons
    
    % plot all trials with all trajectories
    %plot_trajectories(training_data)

    % find the length of the longest trial
    L = 0;
    fprintf("\nFinding the length of longest trial... ");
    t0 = tic; % Begin ticker
    for k = 1:n_trjs
        for n = 1:n_trials    
            l = length(training_data(n,k).spikes(1,:));
            if l > L
                L = l;
            end
        end
    end
    fprintf("Done (%.2f s).\n", toc(t0));
    
    % make all trajectories the same length
    fprintf("Making all trajectories the same length (%d pts)... ", L);
    t0 = tic; % Begin ticker
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
    fprintf("Done (%.2f s).\n", toc(t0));
    
    % calculate the average trajectory
    avg_trjs(n_trjs).handPos = [];
    fprintf("Calculating average trajectories... ");
    t0 = tic; % Begin ticker
    for k = 1:n_trjs
        trj = zeros(2, L);
        for n = 1:n_trials
            for j = 1:length(training_data(n, k).handPos(1, :))
                trj(:, j) = trj(:, j) + training_data(n, k).handPos(1:2, j);
            end
        end
        
        avg_trjs(k).handPos = trj(:, :) / n_trials;
    end
    fprintf("Done (%.2f s).\n", toc(t0));

    % plot average trajectory
    %plot_avg_trajectories(avg_trjs)
    
    % Collect spike rate (density) function for each trial
    fprintf("Collecting spike density function for each trial... ");
    t0 = tic; % Begin ticker
    x = -spike_dist_window:1:spike_dist_window;
    y = normpdf(x, 0, spike_dist_std);
    for n = 1:n_trials
        for k = 1:n_trjs
            training_data(n, k).spikeDist = zeros(n_neuron, L);
            for i = 1:n_neuron
                % Mixture model of spike distribution resembling localised (time-dependent) spike rate, binned at dt = 1 ms
                for l = 1:L
                    if training_data(n, k).spikes(i, l) == 1
                        for j = -spike_dist_window:spike_dist_window
                            if j+l > 0 && j+l < L
                                training_data(n, k).spikeDist(i, l+j) = training_data(n, k).spikes(i, l+j) ...
                                                                      + y(j+spike_dist_window+1);
                            end
                        end
                    end
                end
            end
        end
    end
    fprintf("Done (%.2f s).\n", toc(t0));
    
    % Average spike rate (density) across all trials for each neuron
    avg_spike_rate = zeros(n_trjs, n_neuron, L);
    fprintf("Calculating average spike rate across all trials for each neuron... ");
    t0 = tic; % Begin ticker
    for k = 1:n_trjs
        for i = 1:n_neuron
            for t = 1:L
                spike_sum = 0;
                for n = 1:n_trials
                    spike_sum = spike_sum + training_data(n, k).spikeDist(i, t);
                end
                avg_spike_rate(k, i, t) = spike_sum / n_trials;
            end
        end
    end
    fprintf("Done (%.2f s).\n", toc(t0));
    
    % Linear regression between spike densities and average trajectories
    fprintf("Begin calculation of linear regression weights... ");
    t0 = tic; % Begin ticker
    pos_preds = []; % 7800 x 2
    beta = zeros(n_trjs, n_neuron, 2); % 8 x 98 x 2
    
    features = []; % 98 x 7800
    labels = []; % 1 x 7800
    for k = 1:n_trjs
        pos = [avg_trjs(k).handPos(1, :); avg_trjs(k).handPos(2, :)]'; % 975 x 2
        firing_rate = [squeeze(avg_spike_rate(k, :, :))]; %  98 x 975
        
        features = [features , firing_rate];
        these_labels = ones(1, L) * k;
        labels = [labels , these_labels];
        
        beta_now = lsqminnorm(firing_rate', pos); % 98 x 2
                        
        beta(k, :, :) = beta_now;
                
        pos_pred = firing_rate' * beta_now; % 975 x 2
        pos_preds = [pos_preds; pos_pred]; % 7800 x 2
    end
    fprintf("Done (%.2f s).\n", toc(t0));
        
    model_params.beta = beta;
    
    % Plot prediction
    for k = 1:n_trjs
        
        end_point = k * L;
        
        start_point = (k-1) * L + 1;
        pos_current = pos_preds(start_point:end_point, :);

        x_pos = pos_current(:, 1);
        y_pos = pos_current(:, 2);

        plot(x_pos, y_pos, 'r')
        
        hold on
    end
    axis square
    
    features = features'; % 7800 x 98
    labels = labels';
    Mdl = fitcknn(features,labels, 'NumNeighbors', k_num_neighbours);
    model_params.mdl = Mdl;
    
    accuracy = 0;
    for i = 1:L*n_trjs
        x_in = features(i,:);
        prediction = predict(Mdl, x_in);
        if prediction == labels(i)
            accuracy = accuracy + 1;
        end
    end
    disp(accuracy/(L*n_trjs));
    whos
    figure
end