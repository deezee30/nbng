%%% Team Members: Faillace, Elena; Lazzaroli, Chiara; Lawrence, Kai; Zerkalijs, Deniss
function model_params = positionEstimatorTraining(training_data)
    % Arguments:

    % - training_data:
    %     training_data(n,k)              (n = trial id,  k = reaching angle)
    %     training_data(n,k).trialId      unique number of the trial
    %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
    %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)

    % ... train your model

    % Return Value:

    % - model_params:
    %     single structure containing all the learned parameters of your
    %     model and which can be used by the "positionEstimator" function.
  
    % KNN does not require training
    model_params.trial = training_data;
    model_params.angle = -1;
end