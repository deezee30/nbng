%%% Team Members: Faillace, Elena; Lazzaroli, Chiara; Lawrence, Kai; Zerkalijs, Deniss
function [model_params] = positionEstimatorTraining(training_data)

    data = {}; %data{1} will keep the spike data (that is averaged and times are gone) over angle 1 trials/instances
    [num_trials , num_classes]= size(training_data); %keep size 100,8
    nr_neurons = size(training_data(1,1).spikes,1); %98, this is number of rows in .spike data
    group_size = 300; %we will average these many milliseconds to only keep neuron information
    for class = 1:num_classes %for everz class do the following
       spikes = zeros(num_trials, nr_neurons); %a classes dataset will have 100 ...
       %rows for each trial and 98 variables for each neuron
       for t = 1:num_trials %for 100 instances, fill the spike rows (spike's row is a trial that averages over time)
           spikes(t, :) = mean(training_data(t,class).spikes(:, 1:group_size), 2); %corresponding class dataset's t'th instance's 1:300 times are averaged for each neuron
       end
       data{class} = spikes; %save spikes as this angle's dataset (spikes will be overriden next)
    end
    % train - test split (start with 80/20, then cross validate)
    nr_training = 80; %we can cross validate this easily, discuss with team. Now we just take first 80 instances as training
    train = {}; %every angle will have a train set (the training instances where we know the angle)
    test = {}; %same as above but for test set (last 20 instances of each angle)
    for class = 1:num_classes %for every angle
        train{class} = data{class}(1:nr_training, :); %just take the first 80
        test{class} = data{class}(nr_training+1:num_trials, :); %last 20
    end
    train_target = repelem(1:num_classes, nr_training)'; %not being used rn
    test_target = repelem(1:num_classes, num_trials - nr_training)'; %repelm([1,2,3],2) does 1,1,2,2,3,3, for example

    %train all combos
    models = {}; %for example models{4,7} will keep the SVM that is trained to figure out 4's versus 7's by using only train{4} and train{7}
    times = []; %28 many SVMs', for each we will keep track of the time to train
    combs = nchoosek(1:num_classes,2); %gives every possible 2-combination of angles 1 to 8 (There are 28 many)
    for pair =1:size(combs,1) %for all pair size
        class_a = combs(pair, 1); %take this pair's first elements
        class_b = combs(pair, 2); %take this pair's second elements
        X = [train{class_a}; train{class_b}]; %we should only train on class a and class b instances (80*2 rows and 98 columns because a column is a neuron)
        tic %start counting
        [model] = svmTrain(X, repelem(0:1, nr_training)', 20, @linearKernel, 0.001, 10000); %train SVM. First input is the predictors. Second is true angles that we write, 20 is penalty that we tuned, linear kernel, 0.001 error tolerance that is default, 10,000 iterations over dataset to train svm
        times = [times, toc] %append the time
        models{class_a,class_b} = model; %save the trained SVM to the pair's index in models cell
    end
    
    model_params.model = models;
    
    % Elena's code
    
    [n_trials, n_angles] = size(training_data);
    
    % find average trajectory for each angle
    model_params.trajectories = {};
    for ang = 1:n_angles    
        % make the handPos trajectories all the same length and find average
        traj = training_data(1,ang).handPos;
        for trial = 2:n_trials
            current_traj = training_data(trial,ang).handPos;
            if length(current_traj) < length(traj)
                traj(:, 1:length(current_traj)) = traj(:, 1:length(current_traj)) + current_traj;
                traj(:, 1:length(current_traj)) = traj(:, 1:length(current_traj))/2;
            elseif length(current_traj) > length(traj)
                current_traj(:, 1:length(traj)) = current_traj(:, 1:length(traj)) + traj;
                current_traj(:, 1:length(traj)) = current_traj(:, 1:length(traj))/2;
                traj = current_traj;
            end
        end
        modelParameters.trajectories = [modelParameters.trajectories, traj];
    end
end