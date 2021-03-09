function [x, y] = positionEstimator(test_data, model_params)
% - test_data:
%     test_data(m).trialID
%         unique trial ID
%     test_data(m).startHandPos
%         2x1 vector giving the [x y] position of the hand at the start
%         of the trial
%     test_data(m).decodedHandPos
%         [2xN] vector giving the hand position estimated by your
%         algorithm during the previous iterations. In this case, N is 
%         the number of times your function has been called previously on
%         the same data sequence.
%     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
%     in this case, t goes from 1 to the current time in steps of 20
%     Example:
%         Iteration 1 (t = 320):
%             test_data.trialID = 1;
%             test_data.startHandPos = [0; 0]
%             test_data.decodedHandPos = []
%             test_data.spikes = 98x320 matrix of spiking activity
%         Iteration 2 (t = 340):
%             test_data.trialID = 1;
%             test_data.startHandPos = [0; 0]
%             test_data.decodedHandPos = [2.3; 1.5]
%             test_data.spikes = 98x340 matrix of spiking activity

[n_trials, n_angles] = size(training_data);
[n_neurons, time_lenght] = size(training_data(1,1).spikes);
time_window = 300;

% transform the dataset to spikes count normalized
transformed_data = zeros(n_neurons);
for neu = 1:n_neurons
    transformed_data(neu) =  sum(test_data.spikes(neu, 1:time_window))/time_window;
end

% TODO: find greatest likelihood

% TODO: find trajectory on average trajectory after given time step (length of spikes) 
  
  
  
end