%%% Team Members: Faillace, Elena; Lazzaroli, Chiara; Lawrence, Kai; Zerkalijs, Deniss
function [modelParameters] = positionEstimatorTraining(training_data)

    [n_trials, n_angles] = size(training_data);
    n_neurons = size(training_data(1,1).spikes, 1);
    time_window = 300;
    
    % find average trajectory for each angle
    
    trajectories = {};
    for ang = 1:n_angles    
        % make the handPos trajectories all the same length and find average
        traj = training_data(1,ang).handPos;
        division_count = zeros(1,1500);
        for trial = 2:n_trials
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
        for i = 1:length(traj)
            traj(:,i) = traj(:,i) / division_count(i);
        end
        trajectories = [trajectories, traj];
    end
    modelParameters.trajectories = trajectories;
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    %    SUB ANGLE METHOD     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    % Get the average coordinate of each trajectory
    
    % Contains the angle of the average coord of average trajectory
    cross_angle = zeros(n_angles,1);
    
    % Contains the angle of the average coord of each trajectory
    avg_angle = zeros(n_angles, n_trials);
   
    
    for ang = 1:n_angles
        current_traj = cell2mat(trajectories(ang));
        avg = mean(current_traj');
        cross_angle(ang,:) = atan2(avg(1),avg(2));
        for trial = 1:n_trials
            current_avg = mean(training_data(trial,ang).handPos');
            avg_angle(ang, trial) = atan2(current_avg(1),current_avg(2));
        end
    end
    
    modelParameters.means = zeros(n_angles*2, n_neurons);
    modelParameters.covariances = zeros(n_angles*2, n_neurons, n_neurons);
    cov_condition = 20;
    
    % find average trajectory for each sub angle
    modelParameters.trajectories = {};
    for ang = 1:n_angles    
        traj_1 = training_data(1,ang).handPos;
        traj_2 = training_data(1,ang).handPos;
        division_count_1 = zeros(1,1500);
        division_count_2 = zeros(1,1500);
        
        for trial = 2:n_trials
            current_traj = training_data(trial,ang).handPos; 
            if avg_angle(ang,trial) >= cross_angle(ang)
                for i = 1:length(current_traj)
                    division_count_1(i) = division_count_1(i) + 1;
                end
                if length(current_traj) < length(traj_1)
                    traj_1(:, 1:length(current_traj)) = traj_1(:, 1:length(current_traj)) + current_traj;
                elseif length(current_traj) > length(traj_1)
                    current_traj(:, 1:length(traj_1)) = current_traj(:, 1:length(traj_1)) + traj_1;
                    traj_1 = current_traj;
                end
                
            else
                for i = 1:length(current_traj)
                    division_count_2(i) = division_count_2(i) + 1;
                end
                if length(current_traj) < length(traj_2)
                    traj_2(:, 1:length(current_traj)) = traj_2(:, 1:length(current_traj)) + current_traj;
                elseif length(current_traj) > length(traj_2)
                    current_traj(:, 1:length(traj_2)) = current_traj(:, 1:length(traj_2)) + traj_2;
                    traj_2 = current_traj;
                end
            end
        end
        
        for i = 1:length(traj_1)
            traj_1(:,i) = traj_1(:,i) / division_count_1(i);
        end
        for i = 1:length(traj_2)
            traj_2(:,i) = traj_2(:,i) / division_count_2(i);
        end
        modelParameters.trajectories = [modelParameters.trajectories, traj_1, traj_2];
    end
    
    disp('start to transform dataset');
    for ang = 1:n_angles
        transformed_data_1 = zeros(n_trials, n_neurons);
        transformed_data_2 = zeros(n_trials, n_neurons);
        trial_count_1 = 0;
        trial_count_2 = 0;
        for trial = 1:n_trials
            % discern which sub traj this should take
            if avg_angle(ang, trial) >= cross_angle(ang)
                trial_count_1 = trial_count_1 + 1;
                for neu = 1:n_neurons
                    transformed_data_1(trial_count_1,neu) = sum(training_data(trial, ang).spikes(neu, 1:time_window))/time_window;
                end
            else 
                trial_count_2 = trial_count_2 + 1;
                for neu = 1:n_neurons
                    transformed_data_2(trial_count_2,neu) = sum(training_data(trial, ang).spikes(neu, 1:time_window))/time_window;
                end
            end 
        end
        modelParameters.means(2*ang - 1, :) = mean(transformed_data_1(1:trial_count_1,:)); % 1 x 98
        modelParameters.covariances(2 * ang - 1, :, :) = cov(transformed_data_1(1:trial_count_1,:)) + eye(n_neurons)*cov_condition; % 98 x 98
        modelParameters.means(2*ang, :) = mean(transformed_data_2(1:trial_count_2,:)); % 1 x 98
        modelParameters.covariances(2*ang, :, :) = cov(transformed_data_2(1:trial_count_2,:)) + eye(n_neurons)*cov_condition; % 98 x 98
    end
end