function [train_data,test_data] = sample(data, perc)
% split the data randmly perc% on training set and 1-perc% on test set
% data is matrix with samples on rows and classes on columns
[n_samples, n_categories] = size(data);
n_train = round(n_samples*perc);

train_idx = randsample(n_samples, n_train);
train_data = data(train_idx,:);
test_data = data(setdiff(1:end, train_idx),:);

end

