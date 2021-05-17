function [x, y, newParameters] = positionEstimator(past_current_trial, modelParameters)

    %first let us find the angle by using the trained model
    newParameters = modelParameters; %update later
    group_size = 300; %how many ms of spikes to look
    spikes = mean(past_current_trial.spikes(:, 1:group_size), 2);
    num_classes = 8; %:(
    max_time = size(past_current_trial.spikes, 2); %we want to estimate this time
    combs = nchoosek(1:num_classes,2); %gives every possible 2-combination of angles 1 to 8 (There are 28 many)

%     combs = [   1,2,3,4; 5,6,7,8;
%                 2,3,4,5; 6,7,8,1;
%                 3,4,5,6; 7,8,1,2;
%                 4,5,6,7; 8,1,2,3    ];
            
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
        if (size(position_trial, 2) >= max_time) && norm(modelParameters.olddata(train_trial, decision).handPos(1:2,1) - past_current_trial.startHandPos(1:2,1)) <= 7 %optimise this 5
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
