%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

rng_list = [1, 20, 50, 100, 250, 1000, 2500, 5000, 10000, 50000]; %different rng used to train and test decoder
dists = [1 2 3 4 5 6 7 8 9 10]; %different euclidean distances tested for average trajectory calculations to predict (x,y) coordinates

%Plot figure of RMSE for each euclidean distance

figure
plot(dists, R_Y, "-o", "color", [0.67, 0.22, 1], "MarkerFaceColor", [0.67, 0.22, 1])
xlabel('Euclidean Distance')
ylabel('RMSE')