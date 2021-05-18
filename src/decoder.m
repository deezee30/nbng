% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

% For N runs:
%   seeds       (1xN): Seeds for each run. Set to 0 to generate random seeds
%   data_splits (1xN): Ratio of training:testing data.
function [RMSE, runtime] = decoder(training_data, model, show, seeds, data_splits)

    modelpath = genpath("src/models/" + model);
    addpath(modelpath);
    
    N = length(seeds);
    RMSE = zeros(1, N);
    runtime = zeros(1, N);
    
    for n = 1:N
        
        t0 = tic;
        
        % Set random number generator and generate random permutation
        seed = seeds(n);
        if seed ~= 0
            rng(seed);
        end
        ix = randperm(length(training_data));

        % Select training and testing data
        s = data_splits(n) * 100;
        trainingData = training_data(ix(1:s),:);
        testData = training_data(ix((s+1):end),:);

        fprintf("Testing the continuous position estimator (%i/%i): \n", n, N)

        meanSqError = 0;
        n_predictions = 0;  

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
            
            x0_bar = mean(pos(:, :, :, 1), "all");
            y0_bar = mean(pos(:, :, :, 2), "all");
            
            xline(y0_bar, "color", [.5 .5 .5, .5], "linewidth", 1, "linestyle", "--");
            yline(x0_bar, "color", [.5 .5 .5, .5], "linewidth", 1, "linestyle", "--");
        end

        % Train Model
        modelParameters = positionEstimatorTraining(trainingData);

        for tr=1:size(testData,1)
            fprintf("Decoding block %i/%i\n", tr, size(testData, 1));
            pause(0.001)
            for direc=randperm(8) 
                decodedHandPos = [];

                times=320:20:size(testData(tr,direc).spikes,2);

                for t=times
                    past_current_trial.trialId = testData(tr,direc).trialId;
                    past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
                    past_current_trial.decodedHandPos = decodedHandPos;

                    past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 

                    if nargout("positionEstimator") == 3
                        [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                        modelParameters = newParameters;
                    elseif nargout("positionEstimator") == 2
                        [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
                    end

                    decodedPos = [decodedPosX; decodedPosY];
                    decodedHandPos = [decodedHandPos decodedPos];

                    meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;

                end
                n_predictions = n_predictions+length(times);
                if show
                    hold on
                    plot(decodedHandPos(1,:),decodedHandPos(2,:), "color", [0.7, 0.4, 1]);
                    plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times), ...
                         "color", [0.25, 0.7, 0.6, 0.75])
                end
            end
        end

        if show
            legend(["", "", "Decoded Position", "Actual Position"], ...
                   "location", "south", "orientation", "horizontal")
        end

        RMSE(n) = sqrt(meanSqError/n_predictions);
        runtime(n) = toc(t0);
    
    end

    rmpath(modelpath)

end
