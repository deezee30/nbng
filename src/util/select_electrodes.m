function [sel_idx, sel_fire_rate] = select_electrodes(fire_rate, P)
    selec_var = max(fire_rate, [], 2) - min(fire_rate, [], 2); % selectivity variation across bins
    p = prctile(selec_var, P); % lowest P percentile across all bins
    
    sel_fire_rate = [];
    sel_idx = [];
    for i = 1:size(fire_rate, 1)
        if selec_var(i) >= p
            sel_fire_rate = [sel_fire_rate, fire_rate(i, :)']; % append fire rates
            sel_idx = [sel_idx, i]; % append neuron indices in top percentile
        end
    end
end