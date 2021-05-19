%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

M = 100;
N = 570;

figure
% Chosen neurons of interest based on inspection
neurons = [87, 35, 8, 4];
titles = ["Negatively Responsive Electrode", "Positively Responsive Electrode", ...
          "Non-Responsive Electrode",        "Always Responsive Electrode"];

% Plot rasterplots for each of the neurons in the same figure
for nx = 1:length(neurons)
    n = neurons(nx);
    
    spikes = zeros(M,N);
    for i = 1:M
       spikes(i,:) = trial(i,1).spikes(n,1:N); 
    end
    spikes = spikes == 1;
    varargin = {};
    for j = 1:N
       varargin{j,1} = 1:N; 
    end

    subplot(4, 1, nx)
    [xPoints, yPoints] = raster(spikes, 'PlotType','horzline');
    ylabel('Trial')
    xlabel('Time (ms)')
    title(titles(nx) + " (#" + n + ")")
    xline(300)
end

% neg is 87
% positive 35
% no 8
% always 4