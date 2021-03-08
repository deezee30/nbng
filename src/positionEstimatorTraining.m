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
    %plot_trajectories(training_data)

    % find the length of the longest trial
    L = 0;
    disp("Finding Length of Longest trial");
    for k = 1:n_trjs
        for n = 1:n_trials    
            l = length(training_data(n,k).spikes(1,:));
            if l > L
                L = l;
            end
        end
    end
    disp("Complete");
    
    % make all trajectories the same length
    disp("Making all trajectories the same length");
    for k = 1:n_trjs
        for n = 1:n_trials
            for j = length(training_data(n,k).spikes(1,:)) + 1:L % adjust to max length range
                training_data(n,k).handPos = [training_data(n,k).handPos ...
                                              training_data(n,k).handPos(:, end)];
                training_data(n,k).spikes = [training_data(n,k).spikes ...
                                              training_data(n,k).spikes(:, end)];
            end
        end
    end
    disp("Complete");
    
    % calculate the average trajectory
    avg_trjs(n_trjs).handPos = [];
    disp("Calculating average trajectories");
    for k = 1:n_trjs
        trj = zeros(2,L);
        for n = 1:n_trials
            for j = 1:length(training_data(n,k).handPos(1,:))
                trj(:,j) = trj(:,j) + training_data(n,k).handPos(1:2,j);
            end
        end
        
        avg_trjs(k).handPos = trj(:,:) / n_trials;
    end
    disp("Complete");

    % plot average trajectory
    %plot_avg_trajectories(avg_trjs)
    
    % Collect spike rate (density) function for each trial
    disp("Collecting spike density function for each trial");
    window = 80;
    x = -window:1:window;
    y = normpdf(x, 0, 50);

    for n = 1:n_trials
        for k = 1:n_trjs
            training_data(n, k).spikeDist = zeros(n_neuron, L);
            for i = 1:n_neuron
                % pdf resembles localised (time-dependent) spike rate, binned at dt = 1 ms
                % TODO: Replace normpdf() with actual distribution densities
                % TODO: delete this: training_data(n, k).spikeDist(i, :) = normpdf(1:L, L/2, 100);
                for l = 1:L
                    if training_data(n, k).spikes(i,l) == 1
                        for j = -window:window
                            if j+l > 0 &&  j+l < L
                                training_data(n, k).spikeDist(i,l+j) = training_data(n, k).spikes(i,l+j) + y(j+window+1);
                            end
                        end
                    end
                end
            end
        end
    end
    disp("Complete");
    
    % Average spike rate (density) across all trials for each neuron
    avg_spike_rate = zeros(n_trjs, n_neuron, L);
    disp("Calculating average spike rate across all trials for each neuron");
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
    disp("Complete");
    
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
    y_preds = []; % 7800 x 2
    disp("Begin calculation of linear regression weights");
    for k = 1:n_trjs

        pos = [avg_trjs(k).handPos(1, :); avg_trjs(k).handPos(2, :)]'; % 975 x 2
        % currently firing rate only looks at one neuron, modified
        firing_rate = [squeeze(avg_spike_rate(k, :, :))]; %  98 x 975
        
        beta_now = lsqminnorm(firing_rate', pos); % 1 x 2
                        
        % Append
        beta = [beta; beta_now]; % 8 x 2
                
        y_pred = firing_rate' * beta_now; % 975 x 2
        y_preds = [y_preds; y_pred]; % 7800 x 2
        
    end
    disp("Complete");
        
    modelParameters.beta = beta;
    
    % Plot prediction
    for k = 1:n_trjs
        
        end_point = k * L;
        
        start_point = (k-1) * L + 1;
        y_current = y_preds(start_point:end_point, :);

        x_pos = y_current(:,1);
        y_pos = y_current(:,2);

        plot(x_pos, y_pos, 'r')
        
        hold on
    end
    axis square
    
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       Start of Perceptron-esque Classifier        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % The task of this network is to look at the average firing rates and
    % guess which trajectory it is following
    
    % Okay, here's how we're gonna do this
    % we're basically gonna hard code a multi layer perceptron
    % can I even do it? who knows, but we're gonna try
    % most of it is fine the hard part is the derivatives
    % Hell it'll be hard to do this anyway.
    % We're gonna make a single layer perceptron
    % Usually Perceptron changes a tiny amount equal to the learning
    % rate in the correct direction as that calculates how correct it is
    % We are going to do something similar but instead of binary
    % classification we're doing trajecotry classification
    % the error will be how far off it is from the correct value of
    % trajectory, which I am arbitrarily gonna range from 1 to 8
    % The issue will be that there is little feature extraction but hey
    % that's a problem for future us
    

    % First things first, we need input data, the average firing rate
    input_data = [];
    
    % Next we need labels. These will range from 1 to 8
    labels_data = [];
    
    % Fill those arrays up with data here
    for k = 1:n_trjs
        pos = [avg_trjs(k).handPos(1, :); avg_trjs(k).handPos(2, :)]';
        firing_rate = [squeeze(avg_spike_rate(k, :, :))]; % 98 x 975

        input_data = [input_data, firing_rate]; % 98 x (975*8)
        labels = ones(1, 975) * k;
        labels_data = [labels_data, labels];
    end
    
    input_data = input_data'; % (975*8) x 98
    
    % Weight matrix, 99 and not 98 because we're including bias
    w = randn(98, 8);
    
    error_log = zeros(1, 975 * 8);
    change_log = zeros(1, 975 * 8);
    
    % Shuffle data to avoid overfitting
    newRowOrder = randperm( 975 * 8 );
    newRowOrder = randperm( 975 * 8 );
%     disp(newRowOrder(1:10));
    
    % Pick a learning rate, make it tiny
%     lr = 0.000000001; 
    % Learning rate apparently has no effect on accuracy
%     lr = 0.00005; 
    lr = 0.001;
    for i = 1:(975 * 8)
        index = newRowOrder(i);
        
        x = input_data(index, :); % 1 x 98
        
        % apply a term to account for bias
%         x = [x, 1]; % 1 x 99
%         disp(size(x));
%         disp(size(w));
        label_pred = x * w;
        [M,I] = max(label_pred);
        label_index = labels_data(1,index);
        compare_label = zeros(1,8);
        compare_label(label_index) = 1;
        %disp(compare_label);
        error_log(i) = norm(compare_label - label_pred);
        
        for l_i = 1:8
           if l_i ~= label_index
               compare_label(l_i) = 0.1;
           end
        end
     
        % Update w according to the input value weighted by error and
        % learning rate
%         if i > 4000
%             lr = 0.000001;
%         end

%         correct = compare_label * x;
        if I < label_index
            w_new = w - (compare_label' * x * error_log(i) * lr)';
        else
            w_new = w + (compare_label' * x * error_log(i) * lr)';
        end
        change_log(i) = norm(w_new - w);
        w = w_new;
        disp(label_pred);
    end
    
    whos
    
    figure
    x_coords = [1:1:8*975];
    %plot(x_coords, error_log);
    title("Error Plot");
    figure
    %plot(x_coords, change_log);
    title("How much the weights are changing");
    
    accuracy = 0;
    for i = 1:(975*8)
        x = input_data(i, :); % 1 x 98
%         x = [x, 1]; % 1 x 99
        label_pred = x * w;
%         disp(label_pred);
        [M,I] = max(label_pred);
        if  I == labels_data(1, i)
            accuracy = accuracy + 1;
        end
    end
    
    disp("Final Accuracy");
    disp(accuracy / (975*8));
    modelParameters.w = w;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %       End of Perceptron-esque  Classifier       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure
end