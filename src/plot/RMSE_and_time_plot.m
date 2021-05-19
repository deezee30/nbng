%%% No Brain No Gain: Elena Faillace, Kai Lawrence, Chiara Lazzaroli, Deniss Zerkalijs

rng_list = [1, 20, 50, 100, 250, 1000, 2500, 5000, 10000, 50000];

%Do this for every classifier for 10 different rng to find average RMSE and run time
time=zeros(10,1);
RMSES = zeros(10,1);
for rng_index = 1:size(rng_list,2)
    tic
    RMSE = testFunction_for_students_MTb(trial, "No Brain No Gain", 2013);
    time(rng_index)=toc; %store run time
    RMSES(rng_index,1) = [RMSE];  %store RMSE
end

average_RMSE = mean(RMSES);
atd_RMSE = std(RMSES);
average_time = mean(time);
std_time = std(time);

%Plot histogram of average and standard deviations of RMSE and run time for
%each classifier
y = [average_bayes average_time_bayes; average_knn average_time_knn; average_SVM average_time_SVM; average_SVM_4 average_time_SVM_4; ...
    average_SVM_lin average_time_SVM_lin];
b = bar(y,'FaceColor','flat');
set(b(1), 'FaceColor', [183, 42, 160]/255)
set(b(2), 'FaceColor', [101, 16, 212]/255)
set(b, 'EdgeColor','k')
hold on
set(gca,'xticklabel',{'Bayes','K-nn','SVM-28','SVM-4', 'SVM-4 & lin. reg.'});


ngroups = size(y, 1);
nbars = size(y, 2);
groupwidth = min(0.8, nbars/(nbars + 1.5));
i = 1; %averages
x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
errorbar(x, y(:,i), [std_bayes, std_knn, std_SVM, std_SVM_4, std_SVM_lin], '.');
ylabel('cm')
i = 2; %times
yyaxis right
x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
errorbar(x, y(:,i), [std_time_bayes, std_time_knn, std_time_SVM, std_time_SVM_4, std_time_SVM_lin], '.');
legend('RMSE (cm)', ' Time (s)' ,  'Location', 'NorthWest')
ylabel('s')
hold off
alpha(b,.8)