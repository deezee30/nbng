function plot_avg_trajectories(avg_trjs)

    n_trjs = length(avg_trjs);

    % plot average trajectory
    for i = 1:n_trjs
        x = avg_trjs(i).handPos(1,:);
        y = avg_trjs(i).handPos(2,:);
    
        plot(x, y, "--k")
        grid on
        hold on
    end
end