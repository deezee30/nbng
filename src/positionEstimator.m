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
combs = nchoosek(1:num_classes,2); %gives every possible 2-combination of angles 1 to 8 (There are 28 many)
if max_time == 320  %means we don't have a class yet
    prediction_vector = zeros(1,nchoosek(num_classes, 2)); %for each test instance (20*8) we wil come up with 28 predictions (as we trained 28 pairwise models). Here we save the decisions of each classifier for the corresponding row's trial
    for pair =1:size(combs,1) %back to the combs (which was saving all pairs of 1 to 8)
        class_a = combs(pair, 1); %take first element of combination
        class_b = combs(pair, 2); %take second ...
        pred = svmPredict_nested(modelParameters.model{class_a,class_b}, spikes); %now bring the model of a vs. b, and predict all the instances by using this
        if pred == 0 %if pred is 0 it means this is class a
            pred = class_a;
        else
            pred = class_b;
        end
        prediction_vector(pair) = pred; %the next column vas 'el', so save all this prediction vector as the next column of prediction_vector, so for each of the 160 test instance we have the result of a new a versus b classifier
    end

    decision = mode(prediction_vector); %take mode of every row, as a row is a test trial, and there are 28 votes, so mode is the maximally voted class and we can take it
    newParameters.tune = decision;
else
    decision = modelParameters.tune;
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
%you may delete later
if size(xs,2) == 0
    for train_trial = 1:size(modelParameters.olddata, 1) %
        position_trial = modelParameters.olddata(train_trial, decision).handPos(1:2, :);
        if (size(position_trial, 2) >= max_time)
           xs = [xs, position_trial(1, max_time)];
           ys = [ys, position_trial(2, max_time)];
        end
    end
end
%delete later

if size(xs,2) >= 1
   x = mean(xs);
   y = mean(ys);
elseif size(xs,2) == 0
    x = past_current_trial.decodedHandPos(1,end); %previous decision
    y = past_current_trial.decodedHandPos(2,end); 
end

end