function [x, y, newParameters] = positionEstimator(past_current_trial, modelParameters)
    function pred = svmPredict_nested(model, X)
    %SVMPREDICT returns a vector of predictions using a trained SVM model
    %(svmTrain). 
    %   pred = SVMPREDICT(model, X) returns a vector of predictions using a 
    %   trained SVM model (svmTrain). X is a mxn matrix where there each 
    %   example is a row. model is a svm model returned from svmTrain.
    %   predictions pred is a m x 1 column of predictions of {0, 1} values.
    %

    % Check if we are getting a column vector, if so, then assume that we only
    % need to do prediction for a single example
    if (size(X, 2) == 1)
        % Examples should be in rows
        X = X';
    end

    % Dataset 
    m = size(X, 1);
    p = zeros(m, 1);
    pred = zeros(m, 1);

    if strcmp(func2str(model.kernelFunction), 'linearKernel')
        % We can use the weights and bias directly if working with the 
        % linear kernel
        p = X * model.w + model.b;
    elseif contains(func2str(model.kernelFunction), 'gaussianKernel')
        % Vectorized RBF Kernel
        % This is equivalent to computing the kernel on every pair of examples
        X1 = sum(X.^2, 2);
        X2 = sum(model.X.^2, 2)';
        K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
        K = model.kernelFunction(1, 0) .^ K;
        K = bsxfun(@times, model.y', K);
        K = bsxfun(@times, model.alphas', K);
        p = sum(K, 2);
    else
        % Other Non-linear kernel
        for i = 1:m
            prediction = 0;
            for j = 1:size(model.X, 1)
                prediction = prediction + ...
                    model.alphas(j) * model.y(j) * ...
                    model.kernelFunction(X(i,:)', model.X(j,:)');
            end
            p(i) = prediction + model.b;
        end
    end

    % Convert predictions into 0 / 1
    pred(p >= 0) =  1;
    pred(p <  0) =  0;

    end


    %first let us find the angle by using the trained model
    newParameters = modelParameters; %update later
    group_size = 300; %how many ms of spikes to look
    spikes = mean(past_current_trial.spikes(:, 1:group_size), 2);
    num_classes = 8; %:(
    max_time = size(past_current_trial.spikes, 2); %we want to estimate this time
    
%     combs = [   1,2,3,4; 5,6,7,8;
%                 2,3,4,5; 6,7,8,1;
%                 3,4,5,6; 7,8,1,2;
%                 4,5,6,7; 8,1,2,3    ];
            
    if max_time == 320  %means we don't have a class yet
        
        prediction_vector = zeros(1,nchoosek(num_classes, 2)); %for each test instance (20*8) we wil come up with 28 predictions (as we trained 28 pairwise models). Here we save the decisions of each classifier for the corresponding row's trial
        preds = zeros(4,1);
        for svm_num = 1:4
            % classes now contain the list of classes
            pred = svmPredict_nested(modelParameters.model{svm_num},spikes);
            preds(svm_num) = pred;
        end
        if preds(1) == 0
            if preds(2) == 0
                if preds(3) == 0
                    % 0,0,0,0
                    if preds(4) == 0
                        decision = 4;
                    % 0,0,0,1
                    else
                        decision = 3;
                    end
                else
                    % 0,0,1,0
                    if preds(4) == 0
                        disp('ERROR');
                    % 0,0,1,1
                    else
                        decision = 2;
                    end
                    
                end
            else
                if preds(3) == 0
                    % 0,1,0,0
                    if preds(4) == 0
                        disp('ERROR');
                    % 0,1,0,1
                    else
                        disp('ERROR');
                    end
                else
                    % 0,1,1,0
                    if preds(4) == 0
                        disp('ERROR');
                    % 0,1,1,1
                    else
                        decision = 1;
                    end
                    
                end
            end
            
        else
            if preds(2) == 0
                if preds(3) == 0
                    % 1,0,0,0
                    if preds(4) == 0
                        decision = 5;
                    % 1,0,0,1
                    else
                        disp('ERROR');
                    end
                else
                    % 1,0,1,0
                    if preds(4) == 0
                        disp('ERROR');
                    % 1,0,1,1
                    else
                        disp('ERROR');
                    end
                    
                end
            else
                if preds(3) == 0
                    % 1,1,0,0
                    if preds(4) == 0
                        decision = 6;
                    % 1,1,0,1
                    else
                        disp('ERROR');
                    end
                else
                    % 1,1,1,0
                    if preds(4) == 0
                        decision = 7;
                    % 1,1,1,1
                    else
                        decision = 8;
                    end
                    
                end
            end
        end
        newParameters.prediction = decision;
    else
        decision = modelParameters.prediction;
        if decision == 0
           disp('This cannot be possible, the test instance should start from 320 ms.')    
        end
    end
    
    %now we know that this test instance is an angle of decision
    xs = [];
    ys = [];
    for train_trial = 1:size(modelParameters.olddata, 1) %
        position_trial = modelParameters.olddata(train_trial, decision).handPos(1:2, :);
        if (size(position_trial, 2) >= max_time) && norm(modelParameters.olddata(train_trial, decision).handPos(1:2,1) - past_current_trial.startHandPos(1:2,1)) <= 5 %optimise this 5
           xs = [xs, position_trial(1, max_time)];
           ys = [ys, position_trial(2, max_time)];
        end
    end
    
    if size(xs,2) == 0
        avg_traj = cell2mat(modelParameters.trajectories(decision));
        if max_time < size(avg_traj,2)
            x = avg_traj(1, max_time);
            y = avg_traj(2, max_time);
        else
            x = avg_traj(1, end);
            y = avg_traj(2, end);
        end
    else
       x = mean(xs);
       y = mean(ys);
    end


end