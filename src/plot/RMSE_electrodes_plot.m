%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

% rng_list = [1, 20, 50, 100, 250, 1000, 2500, 5000, 10000, 50000]; %different rng used to train and test decoder
% trial_ = {};
% %different number of electrodes used for the different % of electrodes used
% elec = {[1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98],[1   2   3   4   5   6   7   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  27  28  29  31  32  33  34  35  36  37  39  40  41  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  61  62  63  64  65  66  67  68  69  70  71  72  73  75  77  78  79  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98],[1   2   3   4   5   6   7   9  10  11  12  14  15  16  17  18  19  22  27  29  31  32  33  34  35  36  37  40  41  43  44  45  47  48  50  51  52  53  54  55  56  57  58  59  61  62  63  64  65  66  67  68  69  70  71  72  75  77  78  80  81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98],[1   2   3   4   5   6   7   9  11  12  14  16  17  18  22  27  29  31  32  33  34  35  36  37  40  41  44  45  47  48  50  51  52  54  55  56  58  59  61  63  64  65  66  67  68  69  71  72  75  77  78  80  81  82  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98],[1   2   3   4   5   6   7   9  12  14  16  17  18  22  27  31  33  34  36  37  40  41  44  45  47  50  51  55  56  59  61  63  65  66  67  68  69  72  75  77  78  80  81  82  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98],[1   2   3   4   7  12  18  22  27  31  33  34  36  37  40  41  44  45  47  50  55  56  59  61  63  65  66  67  68  69  75  77  78  80  81  82  84  85  86  87  88  90  91  92  93  94  96  97  98],[2   3   4   7  12  18  22  27  31  33  34  36  40  41  44  45  56  59  65  66  67  68  69  75  77  80  81  84  85  86  87  88  90  91  92  93  96  97  98],[3   4   7  18  22  27  31  33  34  36  41  44  45  56  59  66  68  69  75  77  81  85  86  87  90  91  92  93  97],[3   4   7  22  31  33  34  36  41  45  56  68  69  75  81  85  87  90  91  93],[3   4   7  31  33  36  41  69  81  90]}
% times = zeros(10,10);
% RMSES = zeros(10,10);
% 
% %Calculate the RMSE for each electrode % used and for each different rng
% for k = 1:size(elec,2) 
%     for n = 1:size(rng_list,2)
%         for i = 1:8
%             for j = 1:100
%                 trial_(j,i).trialId = trial(j,i).trialId;
%                 trial_(j,i).spikes = trial(j,i).spikes(elec{1,k}, :);  
%                 trial_(j,i).handPos = trial(j,i).handPos;
%             end
%         end
%         RMSE_graph_ = testFunction_for_students_MTb(trial_, "No Brain No Gain", rng_list(n));
%         RMSES(n,k) = RMSE_graph_;
%     end
% end

%Plot graph of average and standard deviation of RMSE for different % of electrodes used 

electrodes_ = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10];
figure
plot(electrodes_, averages, "Color", [0.67, 0.22, 1])
hold on
patch([electrodes_ fliplr(electrodes_)], [averages+stdevs fliplr(averages-stdevs)], ...
      [0.67, 0.22, 1], "facealpha", 0.2, "edgealpha", 0)
xlabel("% of Electrodes Used")
ylabel("RMSE")

