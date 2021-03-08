function [modelParameters] = positionEstimatorTraining2(training_data)
[n_trials, n_angles] = size(training_data);
[n_neurons, time_lenght] = size(trainingData(1,1).spikes);
train_batches = 300:20:400;

for t = 1:length(train_batches)
    
end

end



function [train_data,test_data] = sample(data, perc)
% split the data randmly perc% on training set and 1-perc% on test set
% data is matrix with samples on rows and classes on columns
[n_samples, n_categories] = size(data);
n_train = round(n_samples*perc);

train_idx = randsample(n_samples, n_train);
train_data = data(train_idx,:);
test_data = data(setdiff(1:end, train_idx),:);

end

