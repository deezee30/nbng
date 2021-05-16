%%% Team Members: Faillace, Elena; Lazzaroli, Chiara; Lawrence, Kai; Zerkalijs, Deniss
function [x, y, new_params] = positionEstimator(test_data, model_params)

    % - test_data:
    %     test_data(m).trialID
    %         unique trial ID
    %     test_data(m).startHandPos
    %         2x1 vector giving the [x y] position of the hand at the start
    %         of the trial
    %     test_data(m).decodedHandPos
    %         [2xN] vector giving the hand position estimated by your
    %         algorithm during the previous iterations. In this case, N is 
    %         the number of times your function has been called previously on
    %         the same data sequence.
    %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
    %     in this case, t goes from 1 to the current time in steps of 20
    %     Example:
    %         Iteration 1 (t = 320):
    %             test_data.trialID = 1;
    %             test_data.startHandPos = [0; 0]
    %             test_data.decodedHandPos = []
    %             test_data.spikes = 98x320 matrix of spiking activity
    %         Iteration 2 (t = 340):
    %             test_data.trialID = 1;
    %             test_data.startHandPos = [0; 0]
    %             test_data.decodedHandPos = [2.3; 1.5]
    %             test_data.spikes = 98x340 matrix of spiking activity
    
    new_params.trial    = model_params.trial;       % Trial data set
    new_params.avg_traj = model_params.avg_traj;    % Average path for each trajectory
    new_params.k_nn     = model_params.k_nn;        % Number of k-nearest neighbours hyperparameter
    new_params.C_coeff  = model_params.C_coeff;     % Distance coefficient for weighted k-nn algorithm
    trial               = new_params.trial;         % Extract trial
    k_nn                = new_params.k_nn;          % Extract k_nn
    C_coeff             = new_params.C_coeff;       % Extract C_coeff
    [T, K]              = size(trial);              % Shape = (no. of trials  x no. of discrete trajectories)
    [N, L]              = size(test_data.spikes);   % Shape = (no. of neurons x no. of time points)
    pred_t0             = tic;                      % Record prediction block timing
    
    % If first iteration use the predictor to calculate the label
    if L == 320
        mean_test = mean(test_data.spikes, 2)';
        
        % Classification of trajectory k, preset of firing rates if it's the first one
        mean_first = zeros(K, N, T);
        for k = 1:K % For each trajectory
            for m = 1:T % For each trial
                mean_first(k, :, m) = mean(trial(m, k).spikes(:, 1:320), 2);
            end
        end
        
        distances = zeros(K, T);
        for k = 1:K % For each trajectory
            for m = 1:T % For each trial
                distances(k, m) = power(sum(abs(power((mean_test-mean_first(k, :, m)), C_coeff))), 1/C_coeff);
            end
        end
        
        [C, T] = size(distances);
        % sort the distance list in ascending order
        [~, I] = sort(reshape(distances', [C * T, 1]));
        
        % populate with the first k_nn distances
        num_nn = zeros(C, 1);
        for i = 1:k_nn % For each neighbour
            num_nn(ceil(I(i)/T)) = num_nn(ceil(I(i)/T)) + 1;
        end
        
        % obtain label by maximising NN
        [~, new_params.angle] = max(num_nn);
    else
        new_params.angle = model_params.angle;
    end
        
    % Now we know that this test instance is an angle of decision
    xs = [];
    ys = [];
    for train_trial = 1:size(trial, 1)
        position_trial = trial(train_trial, new_params.angle).handPos(1:2, :);

        if size(position_trial, 2) >= L && norm(model_params.trial(train_trial, new_params.angle).handPos(1:2, 1) - test_data.startHandPos(1:2, 1)) <= 7
           xs = [xs, position_trial(1, L)];
           ys = [ys, position_trial(2, L)];
        end
    end
    
    if size(xs, 2) == 0
        avg_traj = cell2mat(new_params.avg_traj(new_params.angle));
        if L < size(avg_traj, 2)
            x = avg_traj(1, L);
            y = avg_traj(2, L);
        else
            x = avg_traj(1, end);
            y = avg_traj(2, end);
        end
    else
       x = mean(xs);
       y = mean(ys);
    end
    
    % Save time
    new_params.duration = toc(pred_t0);
end