%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

% For N runs:
%   model    (string): Name of model corresponding to the directory in "src/models/"
%   show        (t/f): Whether or not to show actual pos. vs. decoded pos. plot for each run
%   seeds       (1xN): Seeds for each run. Set to 0 to generate random seeds
%   data_splits (1xN): Ratio of training:testing data.
function [RMSE, runtime] = decoder(training_data, model, show, seeds, data_splits)

    modelpath = genpath("src/models/" + model);
    addpath(modelpath);
    
    R = length(seeds);     % Number of runs in total
    RMSE = zeros(1, R);    % Root mean square errors for each run
    runtime = zeros(1, R); % Runtime for each run
    
    for r = 1:R
        
        % Record time
        t0 = tic;
        
        % Set random number generator and generate random permutation
        seed = seeds(r);
        if seed ~= 0
            rng(seed);
        end
        ix = randperm(length(training_data));

        % Select training and testing data.
        % Cross-validation depends on the training/testing data split ratio.
        s = data_splits(r) * 100;
        trainingData = training_data(ix(1:s),:);
        testData = training_data(ix((s+1):end),:);

        fprintf("Testing the continuous position estimator (%i/%i): \n", r, R)

        meanSqError = 0;
        n_predictions = 0;  

        % Display only if show is enabled
        if show
            figure
            hold on
            axis square
            grid
            
            title("Actual vs. Predicted Trajectories")
            xlabel("x-position (cm)")
            ylabel("y-position (cm)")
            
            % show normals from center point
            [N, K] = size(training_data);
            [I, T] = size(training_data(1, 1).handPos);
            
            pos = zeros(K, N, I, 2);

            % Convert struct to 4D matrix
            for k = 1:K
                for n = 1:N
                    pos(k, n, :, 1) = training_data(n, k).handPos(1, 1);
                    pos(k, n, :, 2) = training_data(n, k).handPos(2, 1);
                end
            end
            
            % Compute and draw starting x and y axes
            x0_bar = mean(pos(:, :, :, 1), "all");
            y0_bar = mean(pos(:, :, :, 2), "all");
            
            xline(y0_bar, "color", [.5 .5 .5, .5], "linewidth", 1, "linestyle", "--");
            yline(x0_bar, "color", [.5 .5 .5, .5], "linewidth", 1, "linestyle", "--");
        end

        % Train model with training data
        modelParameters = positionEstimatorTraining(trainingData);

        for tr = 1:size(testData, 1)
            fprintf("Decoding block %i/%i\n", tr, size(testData, 1));
            pause(0.001)
            
            for direc = randperm(8) 
                decodedHandPos = [];

                % Test with data after arm starts moving
                times = 320:20:size(testData(tr, direc).spikes, 2);

                for t = times
                    past_current_trial.trialId = testData(tr, direc).trialId;
                    past_current_trial.spikes = testData(tr, direc).spikes(:, 1:t); 
                    past_current_trial.decodedHandPos = decodedHandPos;

                    past_current_trial.startHandPos = testData(tr, direc).handPos(1:2, 1); 

                    % Test the model with testing data.
                    % If enabled, feed the model parameters from training onto testing,
                    if nargout("positionEstimator") == 3
                        [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                        modelParameters = newParameters;
                    elseif nargout("positionEstimator") == 2
                        [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
                    end

                    decodedPos = [decodedPosX; decodedPosY];
                    decodedHandPos = [decodedHandPos decodedPos];

                    % Append to mean square error total
                    meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;

                end
                n_predictions = n_predictions + length(times);
                
                % Plot comparison if plotting is enabled
                if show
                    hold on
                    plot(decodedHandPos(1,:),decodedHandPos(2,:), "color", [0.7, 0.4, 1]);
                    plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times), ...
                         "color", [0.25, 0.7, 0.6, 0.75])
                end
            end
        end

        % Show legend if plotting is enabled
        if show
            legend(["", "", "Decoded Position", "Actual Position"], ...
                   "location", "south", "orientation", "horizontal")
        end

        % Store validations
        RMSE(r) = sqrt(meanSqError/n_predictions);
        runtime(r) = toc(t0);
    
    end

    % Remove model from path to prevent future mix-ups
    rmpath(modelpath)

end