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
    
    new_params.trial    = model_params.trial;       % Copy model params to new params
    trial               = new_params.trial;         % Extract trial
    [T, K]              = size(trial);              % Shape = (no. of trials  x no. of discrete trajectories)
    [N, L]              = size(test_data.spikes);   % Shape = (no. of neurons x no. of time points)
    pred_t0             = tic;                      % Record prediction block timing

    % If first iteration use the predictor to calculate the label
    if L == 320
        mean_test = mean(test_data.spikes');
        
        % Classification of angle, preset of firing rates if it's the first one
        % Best Hyperparameters: k=50, coeff=1
        coeff = 1;
        k = 52;
        mean_320 = zeros(K, N, T);
        for ang_test = 1:K
            for te = 1:T
                mean_320(ang_test, :, te) = mean(trial(te, ang_test).spikes(:, 1:320)')';
            end
        end
        
        distances = zeros(K, T);
        for ang = 1:K
            for tr = 1:T
                distances(ang, tr) = power(sum(abs(power((mean_test-mean_320(ang, :, tr)), coeff))), 1/coeff);
            end
        end
        
        [num_class, T] = size(distances);
        [~, I] = sort(reshape(distances', [num_class * T, 1]));
        num_nn = zeros(num_class, 1);
        for i = 1:k
            num_nn(ceil(I(i)/T)) = num_nn(ceil(I(i)/T)) + 1;
        end
        
        [~, new_params.angle] = max(num_nn);
    else
        new_params.angle = model_params.angle;
    end

    window = 20;
    delay = 0;
    coeff_weight = 5;
    k = 33; % Number of nearest neighbours
    coeff = 2;
    eps = 1e-35;
    mean_test = mean(test_data.spikes(:, L-window-delay:L-delay)');
    angle = new_params.angle;

    % Create Firing rate of specific Window and Delay
    mean_window = zeros(T, N);
    mask = ones(T, 1);
    for i = 1:T
        if length(trial(i, angle).spikes(1, :)) >= L
            mean_window(i, :) = mean(trial(i, angle).spikes(:, L-window-delay:L-delay)');
        else
            mask(i) = 0;
        end
    end
    
    k = min(k, sum(mask));
    overlen = false;
    if k == 0
        k = 5;
        overlen = true;
        mask = ones(T, 1);
        for i = 1:T
            mean_window(i, :) = mean(trial(i, angle).spikes(:, end-window-delay:end-delay)');
        end
    end
    
    % Measure k-NN
    distances = ones(T, 1).*1e15;
    for i = 1:T
        if mask(i)
            distances(i) = power(sum(abs(power((mean_test-mean_window(i, :)), coeff))), 1/coeff);
        end
    end
    
    % Calculate Weights for Positions
    [V, I] = sort(distances);
    weights = zeros(k, 1);
    for i = 1:k
        weights(i) = 1/(V(i)^coeff_weight + eps);
    end
    
    weights = weights./sum(weights);
    
    % Calculate weighted Position
    x = 0;
    y = 0;
    if overlen
        for i = 1:k
            x = x + weights(i)*trial(I(i), angle).handPos(1, end);
            y = y + weights(i)*trial(I(i), angle).handPos(2, end);
        end
    else
         for i = 1:k
            x = x + weights(i)*trial(I(i), angle).handPos(1, L);
            y = y + weights(i)*trial(I(i), angle).handPos(2, L);
         end
    end
    
    % Save time
    new_params.duration = toc(pred_t0);
end