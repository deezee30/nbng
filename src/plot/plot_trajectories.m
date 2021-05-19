%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

function plot_trajectories(training_data)
    
    n_trials = size(training_data, 1); % Number of recorded trials
    n_trjs   = size(training_data, 2); % Number of recorded trajectories
    
    % plot all trials with all trajectories
    for i=1:n_trials
        for j=1:n_trjs
            x = training_data(i,j).handPos(1,:);
            y = training_data(i,j).handPos(2,:);
            
            plot(x, y, "y", "LineWidth", 0.1)
            grid on
            hold on
        end
    end
end