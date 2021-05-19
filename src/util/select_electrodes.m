%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

function [sel_idx, sel_fire_rate] = select_electrodes(fire_rate, P)
    % Finds the most selective electrodes based on P percentile and returns them.
    % 
    % The P lowest percentile of electrodes are discarded. The selectivity is calculated based on
    % the difference in maximum and minimum firing rates for each electrode across all stimuli.

    selec_var = max(fire_rate, [], 2) - min(fire_rate, [], 2); % selectivity variation across bins
    p = prctile(selec_var, P); % lowest P percentile across all bins
    
    sel_fire_rate = [];
    sel_idx = [];
    for i = 1:size(fire_rate, 1)
        % store upper-most electrodes
        if selec_var(i) >= p
            sel_fire_rate = [sel_fire_rate, fire_rate(i, :)']; % append fire rates
            sel_idx = [sel_idx, i]; % append neuron indices in top percentile
        end
    end
end