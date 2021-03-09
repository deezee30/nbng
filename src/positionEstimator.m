function [x, y] = positionEstimator(test_data, model_params)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newmodel_params] = positionEstimator(test_data, model_params)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

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
  
  
  
  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
  
    spike_dist_window = 80; % Window of spike density mixture model
    spike_dist_std    = 50; % Standard deviation of spike density mixture model
    x = -spike_dist_window:1:spike_dist_window;
    y = normpdf(x, 0, spike_dist_std);

    n_neuron = 98;
    
    L = size(test_data.spikes, 2);
    
    spikeDist = zeros(n_neuron, L);
    for i = 1:n_neuron
        % Mixture model of spike distribution resembling localised (time-dependent) spike rate, binned at dt = 1 ms
        for l = 1:L
            if test_data.spikes(i, l) == 1
                for j = -spike_dist_window:spike_dist_window
                    if j+l > 0 && j+l < L
                        spikeDist(i, l+j) = test_data.spikes(i, l+j) + y(j+spike_dist_window+1);
                    end
                end
            end
        end
    end
    
    firing_rate = squeeze(spikeDist)'; % L x 98
    
    % Using current average spikes attempt to classify which trajecotry
    for n = 1: n_neuron
        for l = 1:L
            if firing_rate(l,n) == 0
                firing_rate(l,n) = 0.01;
            end
        end
    end
    mdl = model_params.mdl;
    label_pred = predict(mdl, firing_rate);
    
    beta = model_params.beta; % 8 x 98 x 2
    traj_pred = round(mean(label_pred));
    
    beta_label = beta(traj_pred, :, :); % 1 x 98 x 2
    final_pred = firing_rate(end,:) * squeeze(beta_label); % 1 x 2
    x = final_pred(1,1);
    y = final_pred(1,2);
    disp(final_pred);
end