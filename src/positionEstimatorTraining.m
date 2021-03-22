%%% Team Members: Faillace, Elena; Lazzaroli, Chiara; Lawrence, Kai; Zerkalijs, Deniss
function model_params = positionEstimatorTraining(training_data)
    % Arguments:

    % - training_data:
    %     training_data(n,k)              (n = trial id,  k = reaching angle)
    %     training_data(n,k).trialId      unique number of the trial
    %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
    %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)

    % ... train your model

    % Return Value:

    % - model_params:
    %     single structure containing all the learned parameters of your
    %     model and which can be used by the "positionEstimator" function.
  
    num_trials = size(training_data, 1);              % Number of recorded trials
    num_classes   = size(training_data, 2);              % Number of recorded trajectories

    % find average trajectory for each angle
    trajectories = {};
    for ang = 1:num_classes   

        % make the handPos trajectories all the same length and find average
        traj = training_data(1,ang).handPos;
        division_count = zeros(1,1500);
        for trial = 2:num_trials
            current_traj = training_data(trial,ang).handPos;  

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

        for j = 1:length(traj)
            traj(:,j) = traj(:,j) / division_count(j);
        end
        trajectories = [trajectories, traj];

    end

    model_params.trial = training_data;
    model_params.angle = -1;
    model_params.avg_traj = trajectories;
end