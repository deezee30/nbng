function [modelParameters] = positionEstimatorTraining2(training_data)
[n_trials, n_angles] = size(training_data);
[n_neurons, time_lenght] = size(training_data(1,1).spikes);
time_window = 300;

% compute the firing rates
modelParameters.means = zeros(n_angles, n_neurons);
modelParameters.covariances = zeros(n_angles, n_neurons, n_neurons);
transformed_data = zeros(n_angles, n_trials, n_neurons);
cov_condition = 20;

disp('start to transform dataset');
for ang = 1:n_angles
    for trial = 1:n_trials
        for neu = 1:n_neurons
            % sum over the spikes of a neuron for all the time bins
            % normalized with the number of time bins
            transformed_data(ang, trial, neu) = sum(training_data(trial, ang).spikes(neu, 1:time_window))/time_window;
        end
    end
    modelParameters.means(ang, :) = mean(squeeze(transformed_data(ang, :, :)));
    modelParameters.covariances(ang, :, :) = cov(squeeze(transformed_data(ang, :, :))) + eye(n_neurons)*cov_condition;
end

disp('Estimate the accuracy on the training dataset');

% estimate accuracy on the training dataset
count_right = 0;
count_wrong = 0;

% go throug all the training dataset
for real_label = 1:n_angles  
    for sample = 1:n_trials
       % prob of being a class, look for the greatest
       max_prob = 0;
       max_prob_class = 0;
       for pred_label = 1:n_angles
           % just compute the gaussian likelihood
           cov_mat = squeeze(modelParameters.covariances(pred_label, :, :));
           means = modelParameters.means(pred_label, :);
           coef = 1/sqrt((2*pi)^n_neurons * norm(cov_mat));
           prob_class = coef * exp(-0.5*(squeeze(transformed_data(real_label, sample,:)) - means')' * cov_mat^-1 * (squeeze(transformed_data(real_label, sample,:)) - means'));
           
           % check this is the greatest or not, assume the p(class) is the
           % same for all classes
           if prob_class > max_prob
               max_prob_class = pred_label;
               max_prob = prob_class;
           end
       end
       % did the classifier work well?
       if max_prob_class == real_label
           count_right = count_right + 1;
       else
           count_wrong = count_wrong + 1;
       end
       
    end
end

disp('right guesses: ');
disp(count_right);
disp('wrong guesses: ');
disp(count_wrong);
disp('accuracy on the training dataset: ');
disp(count_right/(count_right+count_wrong));
 
end