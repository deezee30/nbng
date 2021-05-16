function discSpikes = disc_bin(data, edges)
    n_neuron = size(data, 1);
    n_bins   = size(edges, 2);
    
    discSpikes = zeros(n_neuron, n_bins);
    for i = 1:n_neuron
        train = data(i, :);
        for bin = 1:(n_bins-1)
            t0 = edges(bin);
            t1 = edges(bin+1);

            discSpikes(i, bin) = nnz(train(t0:t1));
        end
    end
end