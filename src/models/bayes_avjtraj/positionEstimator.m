%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

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

    [n_neurons, time_length] = size(test_data.spikes);
    time_window = 300;

    % transform the dataset to spikes count normalized
    transformed_data = zeros(1,n_neurons);
    for neu = 1:n_neurons
        transformed_data(neu) =  sum(test_data.spikes(neu, 1:time_window))/time_window;
    end

    % prob of being a class, look for the greatest
    max_prob = 0;
    max_prob_class = 0;
    for pred_label = 1:8
       % just compute the gaussian likelihood
       cov_mat = squeeze(model_params.covariances(pred_label, :, :));
       means = model_params.means(pred_label, :);
       coef = 1/sqrt((2*pi)^n_neurons * norm(cov_mat));
       prob_class = coef * exp(-0.5*(transformed_data' - means')' * (cov_mat^-1) * (transformed_data' - means'));

       % check this is the greatest or not, assume the p(class) is the
       % same for all classes
       if prob_class > max_prob
           max_prob_class = pred_label;
           max_prob = prob_class;
       end
    end

    % find trajectory on average trajectory after given time step (length of spikes) 
    avg_traj = cell2mat(model_params.trajectories(max_prob_class));
    if time_length < size(avg_traj,2)
        x = avg_traj(1, time_length);
        y = avg_traj(2, time_length);
    else
        x = avg_traj(1, end);
        y = avg_traj(2, end);
    end

end