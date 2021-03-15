function mi = mutual_info(r, s, nbins)

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
        idx = Pr>0;
        H = -sum(Pr(idx) .* log2(Pr(idx))); % in bits
    end
end