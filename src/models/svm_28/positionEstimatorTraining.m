function modelParameters = positionEstimatorTraining(training_data)
    train = {}; %train{1} will keep the spike data (that is averaged and times are gone) over angle 1 trials/instances
[num_trials , num_classes]= size(training_data); %keep size 80,8
nr_neurons = size(training_data(1,1).spikes,1); %98, this is number of rows in .spike data
group_size = 300; %we will average these many milliseconds to only keep neuron information
blocks = {};
for class = 1:num_classes %for every class do the following
   spikes = zeros(num_trials, nr_neurons); %a classes dataset will have 100 ...
   for t = 1:num_trials %for 80 instances, fill the spike rows (spike's row is a trial that averages over time)
       spikes(t, :) = mean(training_data(t,class).spikes(:, 1:320), 2); %corresponding class dataset's t'th instance's 1:300 times are averaged for each neuron
       temp_blocks = [];
       for neur = 1:nr_neurons
           temp = training_data(t,class).spikes(neur, group_size+1:end); %temporary array);
           temp_blocks(neur, :)  = arrayfun(@(i) mean(temp(i:min(i+20-1,size(temp,2)))),1:20:length(temp)-20+1 + (20-1));
       end
       blocks{t, class} = temp_blocks; %average 20-by-20
   end
   train{class} = spikes; %save spikes as this angle's dataset (spikes will be overriden next)
end

modelParameters.blocks = blocks;

%train all combos
models = {}; %for example models{4,7} will keep the SVM that is trained to figure out 4's versus 7's by using only train{4} and train{7}
combs = nchoosek(1:num_classes,2); %gives every possible 2-combination of angles 1 to 8 (There are 28 many)
for pair =1:size(combs,1) %for all pair size
    class_a = combs(pair, 1); %take this pair's first elements
    class_b = combs(pair, 2); %take this pair's second elements
    X = [train{class_a}; train{class_b}]; %we should only train on class a and class b instances (80*2 rows and 98 columns because a column is a neuron)
    [model] = svmTrain_nested(X, repelem(0:1, size(training_data,1))', 20, @linearKernel, 0.01, 500); %train SVM. First input is the predictors. Second is true angles that we write, 20 is penalty that we tuned, linear kernel, 0.001 error tolerance that is default, 10,000 iterations over dataset to train svm
    models{class_a,class_b} = model; %save the trained SVM to the pair's index in models cell
end
modelParameters.model = models; %send it to the test function
%now all SVMs are trained
% now time to keep all the hand positions to use
modelParameters.olddata = training_data; %use this later to compute location data
modelParameters.tune = 0; %we will tune this later

    trajectories = {};
    for ang = 1:num_classes   
        % make the handPos trajectories all the same length and find average
        traj = training_data(1,ang).handPos;
        division_count = zeros(1,1500);
        for trial = 2:num_trials
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
        for j = 1:length(traj)
            traj(:,j) = traj(:,j) / division_count(j);
        end
        trajectories = [trajectories, traj];
    end
    modelParameters.trajectories = trajectories;
end
