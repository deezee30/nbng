% prepare all data
load("monkeydata_training") %data name is "trial"
data = {}; %data{1} will keep the spike data (that is averaged and times are gone) over angle 1 trials/instances
[num_trials , num_classes]= size(trial); %keep size 100,8
nr_neurons = size(trial(1,1).spikes,1); %98, this is number of rows in .spike data
group_size = 300; %we will average these many milliseconds to only keep neuron information
for class = 1:num_classes %for everz class do the following
   spikes = zeros(num_trials, nr_neurons); %a classes dataset will have 100 ...
   %rows for each trial and 98 variables for each neuron
   for t = 1:num_trials %for 100 instances, fill the spike rows (spike's row is a trial that averages over time)
       spikes(t, :) = mean(trial(t,class).spikes(:, 1:group_size), 2); %corresponding class dataset's t'th instance's 1:300 times are averaged for each neuron
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
%now all SVMs are trained
% time to test

%instance-by-instance
combined_test = []; %just combine all the test data in a single matrix as we should throw a single data matrix to predict function and we shouldn't know about the true angles while testing
for class = 1:num_classes %for all the classes, add the class's training data to combined_test
    combined_test = [combined_test; test{class}];
end
prediction_vector = zeros(size(combined_test,1), nchoosek(num_classes, 2)); %for each test instance (20*8) we wil come up with 28 predictions (as we trained 28 pairwise models). Here we save the decisions of each classifier for the corresponding row's trial
%el = 1; %this will just keep track of the column number, not very important, probably inefficient
for pair =1:size(combs,1) %back to the combs (which was saving all pairs of 1 to 8)
    class_a = combs(pair, 1); %take first element of combination
    class_b = combs(pair, 2); %take second ...
    pred = svmPredict(models{class_a,class_b}, combined_test); %now bring the model of a vs. b, and predict all the instances by using this
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
