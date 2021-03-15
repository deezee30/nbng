function [x, y] = positionEstimator(test_data, modelParameters)

  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
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
  
      % Collect spike rate (density) function for each trial
      
    n_neuron = 98;
    L = modelParameters.L;
    disp("Calculating spike density function");
    window = 80;
    x = -window:1:window;
    y = normpdf(x, 0, 50);
    spikeDist = zeros(n_neuron, L);
    for i = 1:n_neuron
        for l = 1:L
            if test_data.spikes(i,l) == 1
                for j = -window:window
                    if j+l > 0 &&  j+l < L
                        spikeDist(i,l+j) = test_data.spikes(i,l+j) + y(j+window+1);
                    end
                end
            end
        end
    end
    disp("Complete");
    
    % Using current average spikes attempt to classify which trajecotry
    x = spikeDist'; % L x n_neuron
    w = modelParameters.w;
    label_pred = x(end, :) * w;
    [M,I] = max(label_pred);
    
    beta = modelParameters.beta(I); % 8 x 2
    
    % Convert the avg_spike_data into firing rate
    final_firing_rate = [squeeze(avg_spike_rate)]; 
   
    final_pred = final_firing_rate * beta;
    
    x = final_pred(1,1);
    y = final_pred(1,2);
end