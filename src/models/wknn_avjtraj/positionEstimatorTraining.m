%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

function model_params = positionEstimatorTraining(training_data)
    % Arguments:
    % - training_data:
    %     training_data(n,k)              (n = trial id,  k = reaching angle)
    %     training_data(n,k).trialId      unique number of the trial
    %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
    %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
    %
    % ... train your model
    %
    % Return Value:
    % - model_params:
    %     single structure containing all the learned parameters of your
    %     model and which can be used by the "positionEstimator" function.
    
    [T, C] = size(training_data); % Shape = (no. of recorded trials x number of distinct trajectories)

    % find average trajectory for each angle
    trajectories = {};
    for k = 1:C

        % make the handPos trajectories all the same length and find average
        traj = training_data(1, k).handPos;
        division_count = zeros(1, 1000);
        for trial = 2:T
            current_traj = training_data(trial, k).handPos;  

            for i = 1:length(current_traj)
                division_count(i) = division_count(i) + 1;
            end

            if length(current_traj) < length(traj)
                traj(:, 1:length(current_traj)) = traj(:, 1:length(current_traj)) + current_traj;

            elseif length(current_traj) > length(traj)
                current_traj(:, 1:length(traj)) = current_traj(:, 1:length(traj)) + traj;
                traj = current_traj;
            end

        end

        % normalise
        for j = 1:length(traj)
            traj(:, j) = traj(:, j) / division_count(j);
        end
        
        trajectories = [trajectories, traj];
    end

    % set up parameters for prediction
    model_params.trial      = training_data;    % Trial data set
    model_params.avg_traj   = trajectories;     % Average trajectories
    model_params.k_nn       = 28;               % Number of k-nearest neighbours hyperparameter
    model_params.C_coeff    = 1;                % Distance coefficient for weighted k-nn algorithm
end