function [modelParameters] = positionEstimatorTraining(training_data)
[n_trials, n_angles] = size(training_data);
[n_neurons, time_lenght] = size(training_data(1,1).spikes);
time_window = 300;

% compute the firing rates
modelParameters.means = zeros(n_angles, n_neurons);
modelParameters.covariances = zeros(n_angles, n_neurons, n_neurons);
transformed_data = zeros(n_angles, n_trials, n_neurons);
cov_condition = 20;

disp('start to transform dataset');
for ang = 1:n_angles
    for trial = 1:n_trials
        for neu = 1:n_neurons
            % sum over the spikes of a neuron for all the time bins
            % normalized with the number of time bins
            transformed_data(ang, trial, neu) = sum(training_data(trial, ang).spikes(neu, 1:time_window))/time_window;
        end
    end
    modelParameters.means(ang, :) = mean(squeeze(transformed_data(ang, :, :)));
    modelParameters.covariances(ang, :, :) = cov(squeeze(transformed_data(ang, :, :))) + eye(n_neurons)*cov_condition;
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