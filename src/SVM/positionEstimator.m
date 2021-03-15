%%% Team Members: Faillace, Elena; Lazzaroli, Chiara; Lawrence, Kai; Zerkalijs, Deniss
function [x, y, new_params] = positionEstimator(test_data, model_params)

    combs = nchoosek(1:num_classes,2); %gives every possible 2-combination of angles 1 to 8 (There are 28 many)

    %instance-by-instance
    combined_test = []; %just combine all the test data in a single matrix as we should throw a single data matrix to predict function and we shouldn't know about the true angles while testing
    for class = 1:num_classes %for all the classes, add the class's training data to combined_test
        combined_test = [combined_test; test_data{class}];
    end
    prediction_vector = zeros(size(combined_test,1), nchoosek(num_classes, 2)); %for each test instance (20*8) we wil come up with 28 predictions (as we trained 28 pairwise models). Here we save the decisions of each classifier for the corresponding row's trial
    %el = 1; %this will just keep track of the column number, not very important, probably inefficient
    for pair =1:size(combs,1) %back to the combs (which was saving all pairs of 1 to 8)
        class_a = combs(pair, 1); %take first element of combination
        class_b = combs(pair, 2); %take second ...
        pred = svmPredict(model_params{class_a,class_b}, combined_test); %now bring the model of a vs. b, and predict all the instances by using this
        zeros_to_replace = pred==0; %now we have a vector of zeros and ones, so keep track of zeros
        ones_to_replace = pred ==1; %keep track of ones (indices of them)
        pred(zeros_to_replace) = class_a; %now for all 0's replace them with the original angle class_a because this model was comparing class a versus class b, and said '0' if it believes this is class a
        pred(ones_to_replace) = class_b; %same for class b
        prediction_vector(:, pair) = pred'; %the next column vas 'el', so save all this prediction vector as the next column of prediction_vector, so for each of the 160 test instance we have the result of a new a versus b classifier
        %el = el+1; %go to the next column in the next iteration %now I noticed , change el's with pair
    end

    decisions = mode(prediction_vector, 2); %take mode of every row, as a row is a test trial, and there are 28 votes, so mode is the maximally voted class and we can take it
    comparison = [test_target, decisions]; %compare the test target (Truth) to the decisions we gave from one-versus-one SVMS
    error = sum(test_target ~= decisions)/size(combined_test,1); %report the test error
    
    % Elena's code
    
    [~, time_lenght] = size(test_data.spikes);
    
    avg_traj = cell2mat(model_params.trajectories(label_pred));
    if time_lenght < size(avg_traj,2)
        x = avg_traj(1, time_lenght);
        y = avg_traj(2, time_lenght);
    else
        x = avg_traj(1, end);
        y = avg_traj(2, end);
    end
end