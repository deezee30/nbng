function raster_plot(train, set_title, set_ylabel)
    % Plots events (m x n), where:
    %   m = x axis (time) [ms]
    %   n = y axis (trial) [#]

    n_trials   = size(train, 1);
    n_time_pts = size(train, 2);
    t_span     = 1:1:n_time_pts;
    
    figure
    colors = colourise(n_trials);
    colororder(colors)
    
    for n_idx = 1:n_trials
        n_trains = train(n_idx, :);
        scatter(t_span(n_trains==1), n_trains(n_trains==1)*n_idx, 100, "marker", ".")
        hold all
    end
    
    title(set_title)
    ylabel(set_ylabel)
    xlabel("Time, t (ms)")
    
    function order =  colourise(n_neurons)
        % Colourises data to make it distinguishable on the raster plot.

        order = zeros(n_neurons, 3);

        for n = 1:n_neurons
            order(n, 1) = abs(n/n_neurons);
            order(n, 2) = abs(0.5-n/n_neurons);
            order(n, 3) = abs(1-n/n_neurons);
        end
    end
end