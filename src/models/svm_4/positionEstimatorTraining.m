function [modelParameters] = positionEstimatorTraining(training_data)
    function sim = linearKernel(x1, x2)
    %LINEARKERNEL returns a linear kernel between x1 and x2
    %   sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
    %   and returns the value in sim

    % Ensure that x1 and x2 are column vectors
    x1 = x1(:); x2 = x2(:);

    % Compute the kernel
    sim = x1' * x2;  % dot product

    end
    function sim = gaussianKernel(x1, x2, sigma)
    %RBFKERNEL returns a radial basis function kernel between x1 and x2
    %   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    %   and returns the value in sim

    % Ensure that x1 and x2 are column vectors
    x1 = x1(:); x2 = x2(:);

    sim = exp(-(norm(x1 - x2) ^ 2) / (2 * (sigma ^ 2)));

    end

    function [model] = svmTrain_nested(X, Y, C, kernelFunction, ...
                                tol, max_passes) %copying the whole svm function
    %SVMTRAIN Trains an SVM classifier using a simplified version of the SMO 
    %algorithm. 
    %   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
    %   SVM classifier and returns trained model. X is the matrix of training 
    %   examples.  Each row is a training example, and the jth column holds the 
    %   jth feature.  Y is a column matrix containing 1 for positive examples 
    %   and 0 for negative examples.  C is the standard SVM regularization 
    %   parameter.  tol is a tolerance value used for determining equality of 
    %   floating point numbers. max_passes controls the number of iterations
    %   over the dataset (without changes to alpha) before the algorithm quits.
    %
    % Note: This is a simplified version of the SMO algorithm for training
    %       SVMs. In practice, if you want to train an SVM classifier, we
    %       recommend using an optimized package such as:  
    %
    %           LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
    %           SVMLight (http://svmlight.joachims.org/)
    %
    %

    if ~exist('tol', 'var') || isempty(tol)
        tol = 1e-3;
    end

    if ~exist('max_passes', 'var') || isempty(max_passes)
        max_passes = 5;
    end

    % Data parameters
    m = size(X, 1);
    n = size(X, 2);

    % Map 0 to -1
    Y(Y==0) = -1;

    % Variables
    alphas = zeros(m, 1);
    b = 0;
    E = zeros(m, 1);
    passes = 0;
    eta = 0;
    L = 0;
    H = 0;

    % Pre-compute the Kernel Matrix since our dataset is small
    % (in practice, optimized SVM packages that handle large datasets
    %  gracefully will _not_ do this)
    % 
    % We have implemented optimized vectorized version of the Kernels here so
    % that the svm training will run faster.
    if strcmp(func2str(kernelFunction), 'linearKernel')
        % Vectorized computation for the Linear Kernel
        % This is equivalent to computing the kernel on every pair of examples
        K = X*X';
    elseif contains(func2str(kernelFunction), 'gaussianKernel')
        % Vectorized RBF Kernel
        % This is equivalent to computing the kernel on every pair of examples
        X2 = sum(X.^2, 2);
        K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
        K = kernelFunction(1, 0) .^ K;
    else
        % Pre-compute the Kernel Matrix
        % The following can be slow due to the lack of vectorization
        K = zeros(m);
        for i = 1:m
            for j = i:m
                 K(i,j) = kernelFunction(X(i,:)', X(j,:)');
                 K(j,i) = K(i,j); %the matrix is symmetric
            end
        end
    end

    % Train
    dots = 12;
    while passes < max_passes

        num_changed_alphas = 0;
        for i = 1:m

            % Calculate Ei = f(x(i)) - y(i) using (2). 
            % E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
            E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);

            if ((Y(i)*E(i) < -tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0))

                % In practice, there are many heuristics one can use to select
                % the i and j. In this simplified code, we select them randomly.
                j = ceil(m * rand());
                while j == i  % Make sure i \neq j
                    j = ceil(m * rand());
                end

                % Calculate Ej = f(x(j)) - y(j) using (2).
                E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);

                % Save old alphas
                alpha_i_old = alphas(i);
                alpha_j_old = alphas(j);

                % Compute L and H by (10) or (11). 
                if (Y(i) == Y(j))
                    L = max(0, alphas(j) + alphas(i) - C);
                    H = min(C, alphas(j) + alphas(i));
                else
                    L = max(0, alphas(j) - alphas(i));
                    H = min(C, C + alphas(j) - alphas(i));
                end

                if (L == H)
                    % continue to next i. 
                    continue;
                end

                % Compute eta by (14).
                eta = 2 * K(i,j) - K(i,i) - K(j,j);
                if (eta >= 0),
                    % continue to next i. 
                    continue;
                end

                % Compute and clip new value for alpha j using (12) and (15).
                alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;

                % Clip
                alphas(j) = min (H, alphas(j));
                alphas(j) = max (L, alphas(j));

                % Check if change in alpha is significant
                if (abs(alphas(j) - alpha_j_old) < tol)
                    % continue to next i. 
                    % replace anyway
                    alphas(j) = alpha_j_old;
                    continue;
                end

                % Determine value for alpha i using (16). 
                alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));

                % Compute b1 and b2 using (17) and (18) respectively. 
                b1 = b - E(i) ...
                     - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                     - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
                b2 = b - E(j) ...
                     - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                     - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

                % Compute b by (19). 
                if (0 < alphas(i) && alphas(i) < C)
                    b = b1;
                elseif (0 < alphas(j) && alphas(j) < C)
                    b = b2;
                else
                    b = (b1+b2)/2;
                end

                num_changed_alphas = num_changed_alphas + 1;

            end

        end

        if (num_changed_alphas == 0)
            passes = passes + 1;
        else
            passes = 0;
        end

        dots = dots + 1;
        if dots > 78
            dots = 0;
        end
        if exist('OCTAVE_VERSION')
            fflush(stdout);
        end
    end

    % Save the model
    idx = alphas > 0;
    model.X= X(idx,:);
    model.y= Y(idx);
    model.kernelFunction = kernelFunction;
    model.b= b;
    model.alphas= alphas(idx);
    model.w = ((alphas.*Y)'*X)';

    end

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