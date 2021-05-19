%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

function mi = mutual_info(r, s, nbins)
    % Compute mutual information between stimulus and response, given bins.

    [Cr, ~] = histcounts(r, nbins);
    [Cs, ~] = histcounts(s, nbins);
    [Crs, ~, ~] = histcounts2(r, s, nbins);
    
    % Normalisation to give a probability distribution
    Pr = Cr / sum(Cr);
    Ps = Cs / sum(Cs);
    Prs = Crs / sum(Crs);

    Hr = entropy(Pr);
    Hs = entropy(Ps);
    Hrs = entropy(Prs);

    mi = Hr + Hs - Hrs; % in bits
    
    function H = entropy(Pr)
        % Compute shannon entropy of events.
        
        idx = Pr>0;
        H = -sum(Pr(idx) .* log2(Pr(idx))); % in bits
    end
end