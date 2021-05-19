%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

function [modelParameters] = positionEstimatorTraining(training_data)
    % Arguments:
    % - training_data:
    %     training_data(n,k)              (n = trial id,  k = reaching angle)
    %     training_data(n,k).trialId      unique number of the trial
    %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
    %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
    %
    % ... train your model
    %
    % Return Value:
    % - model_params:
    %     single structure containing all the learned parameters of your
    %     model and which can be used by the "positionEstimator" function.
    
    train = {}; %train{1} will keep the spike data (that is averaged and times are gone) over angle 1 trials/instances
    [num_trials , num_classes]= size(training_data); %keep size 80,8
    nr_neurons = size(training_data(1,1).spikes,1); %98, this is number of rows in .spike data
    group_size = 320; %we will average these many milliseconds to only keep neuron information
    
    for class = 1:num_classes %for everz class do the following
       spikes = zeros(num_trials, nr_neurons); %a classes dataset will have 100 ...
       %rows for each trial and 98 variables for each neuron
       for t = 1:num_trials %for 80 instances, fill the spike rows (spike's row is a trial that averages over time)
           spikes(t, :) = mean(training_data(t,class).spikes(:, 1:group_size), 2); %corresponding class dataset's t'th instance's 1:300 times are averaged for each neuron
       end
       train{class} = spikes; %save spikes as this angle's dataset (spikes will be overriden next)
    end

    %train all combos
    models = {}; %for example models{4,7} will keep the SVM that is trained to figure out 4's versus 7's by using only train{4} and train{7}
    times = []; %28 many SVMs', for each we will keep track of the time to train
    
    
    % Instead of going through every possible comparison we only really
    % need four svsms. These will separate between classes:
    %           0               1
    % SVM 1: [1,2,3,4] and [5,6,7,8]
    % SVM 2: [2,3,4,5] and [6,7,8,1]
    % SVM 3: [3,4,5,6] and [7,8,1,2]
    % SVM 4: [4,5,6,7] and [8,1,2,3]
    % By using the 0 or 1 output of each we will know the prediction. For
    % example, if the SVMs output [0,0,0,1] it will be 3. There are
    % impossible cases such as [1,0,1,0]. Basically values will always be
    % adjacent, only 0s then all 1s or only 1s then all 0s.
    % To implement this all we really need to do is modify the input X to 
    % include the train{} data of all the classes 
    
    % In this function X contains two 50 x 98 matrices appended next to
    % each other. So to allow for all the other data we need two (50*4) x
    % 98 matrices
%     combs = [   1,2,3,4; 5,6,7,8;
%                 2,3,4,5; 6,7,8,1;
%                 3,4,5,6; 7,8,1,2;
%                 4,5,6,7; 8,1,2,3    ];
    
    % In this new 7 SVM version which we're testing to see if it decreases
    % error, we need to basically have splits that go:
           
    combs = [   1,2,3,4; 5,6,7,8;
                2,3,4,5; 6,7,8,1;
                3,4,5,6; 7,8,1,2;
                4,5,6,7; 8,1,2,3    ];

    for svm_num = 1:4
        % classes now contain the list of classes
        classes_a = combs(2*svm_num - 1,:)'; %take this pair's first elements
        classes_b = combs(2*svm_num,:)'; %take this pair's second elements
        train_a = [];
        train_b = [];
        for i = 1:4
            train_a = [ train_a , train{classes_a(i)}' ];
            train_b = [ train_b , train{classes_b(i)}' ];
        end
        train_a = train_a';
        train_b = train_b';

        X = [train_a; train_b]; %we should only train on class a and class b instances (80*2 rows and 98 columns because a column is a neuron)
        tic %start counting
        [model] = svmTrain_nested(X, repelem(0:1, size(training_data,1) * 4)', 20, @linearKernel, 0.01, 500); %train SVM. First input is the predictors. Second is true angles that we write, 20 is penalty that we tuned, linear kernel, 0.001 error tolerance that is default, 10,000 iterations over dataset to train svm
        times = [times, toc]; %append the time
        models{svm_num} = model; %save the trained SVM to the pair's index in models cell
    end
    modelParameters.model = models; %send it to the test function
    %now all SVMs are trained
    % now time to keep all the hand positions to use
    
    modelParameters.olddata = training_data; %use this later to compute location data
    modelParameters.runtime = sum(times);
    modelParameters.prediction = 0; %we will tune this later    
    
    % find average trajectory for each angle
    trajectories = {};
    for ang = 1:num_classes   
        % make the handPos trajectories all the same length and find average
        traj = training_data(1,ang).handPos;
        division_count = zeros(1,1500);
        for trial = 2:num_trials
            current_traj = training_data(trial,ang).handPos;  
            for i = 1:length(current_traj)
                division_count(i) = division_count(i) + 1;
            end
            if length(current_traj) < length(traj)
                traj(:, 1:length(current_traj)) = traj(:, 1:length(current_traj)) + current_traj;
            elseif length(current_traj) > length(traj)
                current_traj(:, 1:length(traj)) = current_traj(:, 1:length(traj)) + traj;
                traj = current_traj;
            end
        end
        for j = 1:length(traj)
            traj(:,j) = traj(:,j) / division_count(j);
        end
        trajectories = [trajectories, traj];
    end
    modelParameters.trajectories = trajectories;
end