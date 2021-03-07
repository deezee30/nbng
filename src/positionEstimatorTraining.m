function [modelParameters] = positionEstimatorTraining(training_data)
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
    % - modelParameters:
    %     single structure containing all the learned parameters of your
    %     model and which can be used by the "positionEstimator" function.
    
    n_trials = size(training_data, 1);              % Number of recorded trials
    n_trjs   = size(training_data, 2);              % Number of recorded trajectories
    n_neuron = size(training_data(1,1).spikes, 1);  % Number of neurons
    
    % plot all trials with all trajectories
    plot_trajectories(training_data)

    % find the length of the longest trial
    L = 0;
    for k = 1:n_trjs
        for n = 1:n_trials    
            l = length(training_data(n,k).spikes(1,:));
            if l > L
                L = l;
            end
        end
    end
    fprintf("... ")
    
    % make all trajectories the same length
    for k = 1:n_trjs
        for n = 1:n_trials
            for j = length(training_data(n,k).spikes(1,:)) + 1:L % adjust to max length range
                training_data(n,k).handPos = [training_data(n,k).handPos ...
                                              training_data(n,k).handPos(:, end)];
            end
        end
    end
    fprintf("... ")

    % calculate the average trajectory
    avg_trjs(n_trjs).handPos = [];
    for k = 1:n_trjs
        trj = zeros(2,L);
        for n = 1:n_trials
            for j = 1:length(training_data(n,k).handPos(1,:))
                trj(:,j) = trj(:,j) + training_data(n,k).handPos(1:2,j);
            end
        end
        
        avg_trjs(k).handPos = trj(:,:) / n_trials;
    end
    fprintf("... ")

    % plot average trajectory
    plot_avg_trajectories(avg_trjs)
    
    % Collect spike rate (density) function for each trial
    for n = 1:n_trials
        for k = 1:n_trjs
            for i = 1:n_neuron
                % pdf resembles localised (time-dependent) spike rate, binned at dt = 1 ms
                % TODO: Replace normpdf() with actual distribution densities
                training_data(n, k).spikeDist(i, :) = normpdf(1:L, L/2, 100);
            end
        end
    end
    fprintf("... ")
    
    % Average spike rate (density) across all trials for each neuron
    avg_spike_rate = zeros(n_trjs, n_neuron, L);
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
    fprintf("... ")
    
%     % information theory
%     
%     % the response over the same trial, different neurons (ideally high)
%     for k = 1:n_trjs
%         for n = 1:n_trials
%             % stimulus = spike train of 1 neuron averaged across
%             % response = pos x, pos y
%         end
%     end
%     
%     % the response over the same neurons, different trials (ideally low)
%     for k = 1:n_trjs
%         for i = 1:n_neuron
%             
%         end
%     end
    
    % Linear regression between spike densities and average trajectories
    beta = []; % 8 x 2
    y_preds = []; % 6336 x 2
    for k = 1:n_trjs

        pos = [avg_trjs(k).handPos(1, :); avg_trjs(k).handPos(2, :)]';
        firing_rate = [squeeze(avg_spike_rate(k, 1, :))];
        
        beta_now = lsqminnorm(firing_rate, pos); % 1 x 2
                        
        % Append
        beta = [beta; beta_now]; % 8 x 2
                
        y_pred = firing_rate * beta_now; % 792 x 2
        y_preds = [y_preds; y_pred]; % 6336 x 2
        
    end
    fprintf("... ")
        
    modelParameters.beta = beta;
    
    % Plot prediction
    figure
    for k = 1:n_trjs
        
        end_point = k * L;
        
        start_point = (k-1) * L + 1;
        y_current = y_preds(start_point:end_point, :);

        x_pos = y_current(:,1);
        y_pos = y_current(:,2);

        plot(x_pos, y_pos, 'r')
        
        hold on
    end
    
    whos
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       Start of Neural Network Code        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     % Define shape of network (basic MLP but require linear activation on
%     % final layer due to coordinate data)
%     
%     network_input_data = [];
%     network_output_data = [];
%     for k = 1:n_trjs
%         pos = [avg_trjs(k).handPos(1, :); avg_trjs(k).handPos(2, :)]';
%         firing_rate = [squeeze(avg_spike_rate(k, 1, :))];
% 
%         network_input_data = [network_input_data; firing_rate];
%         network_output_data = [network_output_data; pos];
%     end
%     
%     layers = [ ...
%         featureInputLayer(98)
%         fullyConnectedLayer(500)
%         tanhLayer
%         fullyConnectedLayer(50)
%         fullyConenctedLayer(2)
%         regressionLayer
%         ];
%     
%     % Parameters for training
%     options = trainingOptions('sgdm', 'MaxEpochs', 10, 'InitialLearnRate', 0.001, 'Plots', 'training-progress');
%     
%     % Train the network
%     net = trainNetwork(network_input_data, network_output_data, layers, options);
%     
%     % Get predictions
%     coord_prediciton = predict(net, input_data);
%     
%     % Plot predictions
%     figure
%     x_pos = coord_prediction(:,1);
%     y_pos = coord_prediction(:,2);
%     plot(x_pos, y_pos, 'r')
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       End of Neural Network Code        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end