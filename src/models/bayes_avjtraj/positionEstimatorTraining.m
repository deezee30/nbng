function [modelParameters] = positionEstimatorTraining2(training_data)
    [n_trials, n_angles] = size(training_data);
    [n_neurons, time_length] = size(training_data(1,1).spikes);
    time_window = 300; % do not use the entire length of spike train - just the planning part (first 300 ms)

    % compute the firing rates
    modelParameters.means = zeros(n_angles, n_neurons); % (8 x 98)
    modelParameters.covariances = zeros(n_angles, n_neurons, n_neurons); % (8 x 98 x 98)
    transformed_data = zeros(n_angles, n_trials, n_neurons);
    cov_condition = 20;

    % transform the dataset into a format that makes it easier to compute the covariance matrices
    disp('start to transform dataset');
    for ang = 1:n_angles
        for trial = 1:n_trials
            for neu = 1:n_neurons
                % sum over the spikes of a neuron for all the time bins
                % normalized with the number of time bins
                transformed_data(ang, trial, neu) = sum(training_data(trial, ang).spikes(neu, 1:time_window))/time_window;
            end
        end
        modelParameters.means(ang, :) = mean(squeeze(transformed_data(ang, :, :))); % (8 x 98)
        modelParameters.covariances(ang, :, :) = cov(squeeze(transformed_data(ang, :, :))) + eye(n_neurons)*cov_condition; % (8 x 98 x 98)
    end

    % find average trajectory for each angle
    modelParameters.trajectories = {};
    for ang = 1:n_angles    
        % make the handPos trajectories all the same length and find average
        traj = training_data(1,ang).handPos;
        for trial = 2:n_trials
            current_traj = training_data(trial,ang).handPos;
            if length(current_traj) < length(traj)
                traj(:, 1:length(current_traj)) = traj(:, 1:length(current_traj)) + current_traj;
                traj(:, 1:length(current_traj)) = traj(:, 1:length(current_traj))/2;
            elseif length(current_traj) > length(traj)
                current_traj(:, 1:length(traj)) = current_traj(:, 1:length(traj)) + traj;
                current_traj(:, 1:length(traj)) = current_traj(:, 1:length(traj))/2;
                traj = current_traj;
            end
        end
        modelParameters.trajectories = [modelParameters.trajectories, traj];
    end
end