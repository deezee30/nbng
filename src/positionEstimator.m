function [x, y, new_params] = positionEstimator(test_data, model_params)

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
    n_neuron          = 98; % Number of neurons
    bin_size = model_params.bin_size;
    
    L = size(test_data.spikes, 2);
    n_bins = fix(L/bin_size) + 1;
%     disp(n_bins);

    edges = fix(linspace(1, L, n_bins));

    % Binning spikes into n_bins bins to save processing time
    discSpikes = disc_bin(test_data.spikes(:, :), edges);
        
    % Collect spike rate (density) function for each trial
    x_window = -spike_dist_window:1:spike_dist_window;
    y_gauss = normpdf(x_window, 0, spike_dist_std);

    spikeDist = zeros(n_neuron, n_bins);
    for i = 1:n_neuron
        % Mixture model of spike distribution resembling localised (time-dependent)
        % spike rate, binned at (L/n_bins) widths.
        for bin = 1:n_bins
            if discSpikes(i, bin) > 0
                for j = -spike_dist_window:spike_dist_window
                    if j+bin > 0 && j+bin < n_bins
                        spikeDist(i, bin+j) = discSpikes(i, bin+j) + y_gauss(j+spike_dist_window+1);
                    end
                end
            end
        end
    end
%     disp(spikeDist(1,:));
    noise = randn(n_neuron, n_bins) / 5;
    for i = 1:n_neuron
        for bin = 1:n_bins
            if spikeDist(i, bin) > 0
                n = noise(i, bin);
                if  noise(i, bin) < 0
                    n = n * -1;
                end
                spikeDist(i,bin) = spikeDist(i,bin) + n;
            end
        end
    end
%     spikeDist = spikeDist + noise;
    
    % Using current average spikes attempt to classify which trajecotry
    % If first iteration use the predictor to calculate the label
    label_pred = 0;
    new_params = model_params;
    if L == 320
        label_pred = mode(predict(model_params.mdl, spikeDist'));
        new_params.label = label_pred;
    % If subsequent loops use the recorded value in model_params
    else
        label_pred = model_params.label;
    end
    
    beta = model_params.beta; % 8 x 98 x 2
    
    beta_label = beta(label_pred, :, :); % 1 x 98 x 2
    final_pred = (spikeDist') * squeeze(beta_label); % n_bins x 2
    final_pred = final_pred;
    x = final_pred(end-1,1);
    y = final_pred(end-1,2);
    if (x^2 + y^2) > 150^2
        x = 0;
        y = 0;
    end
%     disp(L);
%     plot_x = final_pred(:,1);
%     plot_y = final_pred(:,2);
%     plot(plot_x, plot_y)
%     hold on
%     disp(final_pred);
%     x_coord = [1:1:n_bins];
%     plot(x_coord, spikeDist(1,:), 'b')
%     hold on
%     figure
end