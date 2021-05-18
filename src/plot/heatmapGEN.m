load('SVM_4_RMSE_LOG.mat')

xvalues = {'1', '20', '42', '50', '75', '100', '250', '327', '500', '750', '1000', '2500', '5000', '6847', '7500', '10000', '21278', '25000', '50000', '75000', '88178', '100000', 'Mean across Seeds'};
yvalues = {'10/90','15/85', '20/80', '25/75', '30/70', '35/65', '40/60', '45/55', '50/50', '55/45', '60/40', '65/35', '70/30', '75/25', '80/20', '85,15', '90/10', '95/15', 'Mean across Splits'};
new_column = ones(18, 1);
new_row = ones(1, 22);

disp(size(RMSE_log))
for i = 1:18
    cell_avg = 0;
    cell_count = 0;
    for j = 1:22
        if RMSE_log(i,j) < 50
            cell_avg = cell_avg + RMSE_log(i,j);
            cell_count = cell_count + 1;
        end
    end
    new_column(i) = cell_avg/cell_count;
end

for i = 1:22
    cell_avg = 0;
    cell_count = 0;
    for j = 1:18
        if RMSE_log(j,i) < 50
            cell_avg = cell_avg + RMSE_log(j,i);
            cell_count = cell_count + 1;
        end
    end
    new_row(1, i) = cell_avg/cell_count;
end

% 
% old_new_column = [ mean(RMSE_log, 2) ; NaN];
% old_new_row = mean(RMSE_log);

% disp(size(RMSE_log))
% disp(new_row)
% disp(old_new_row)
% disp(new_column)
% disp(old_new_column)
new_column = [new_column ; NaN];
RMSE_log = [RMSE_log; new_row];
RMSE_log = [RMSE_log, new_column];
figure(1)
h = heatmap(xvalues, yvalues, RMSE_log);
colormap cool
% h = heatmap(RMSE_log);

h.XLabel = 'Random Seed';
h.YLabel = 'Training/Testing Data Size';
h.FontSize = 18;

figure(2)
hold on
x = categorical({'10/90', '20/80', '30/70', '40/60', '50/50', '60/40', '70/30', '80/20', '90/10'});
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18];

avgs = new_column(1:end - 1)';

% plot(x,avgs)
std_dev = zeros(1,18);

meta_stat = [25, 27, 29, 31, 33, 35, 37, 39];

for i = 1:18
    considered_array = zeros(22,1);
    for j = 1:22
        if RMSE_log(i,j) < 50
            considered_array(j) = RMSE_log(i,j);
%             considered_array = [considered_array RMSE_log(i,j)];
%         else
%             r = randsample([1,2,3,4,5,6,7,8], 1);
%             decision = meta_stat(r);
%             considered_array = [considered_array , decision];
        end
    end
    
    std_dev(1,i) = std(nonzeros(considered_array));
%     std_dev(1,i) = std(RMSE_log(i,:));
end
plot(x,avgs,'color', [84/256, 214/256, 201/256], 'LineWidth',1.5)
patch([x fliplr(x)], [avgs - std_dev fliplr(avgs + std_dev)], [84/256 214/256 201/256], 'LineStyle', 'none');

% figure(3)
% 
% err_high = zeros(1,9);
% err_low = zeros(1,9);
% for i = 1:9
%     max = -1;
%     min = 9999;
%     for j = 1:10
%         if RMSE_log(i,j) < 50
%             if min > RMSE_log(i,j)
%                 min = RMSE_log(i,j);
%             end
%             if max < RMSE_log(i,j)
%                 max = RMSE_log(i,j);
%             end
%         end
%     end
%     err_high(1,i) = max - new_column(i);
%     err_low(1,i) = new_column(i) - min;
% end
% x2 = [2 5; 2 5; 8 8];
% y2 = [4 0; 8 2; 4 0];
% patch(x2,y2,'green')
% 
% er = errorbar(x,new_column(1:end-1),err_low,err_high);    
% er.Color = [0 0 0];                            
% er.LineStyle = 'none';
xlabel('Training/Test Dataset Size', 'Fontsize', 14)
ylabel('RMSE', 'Fontsize', 14)
xticklabels({'10/90','15/85', '20/80', '25/75', '30/70', '35/65', '40/60', '45/55', '50/50', '55/45', '60/40', '65/35', '70/30', '75/25', '80/20', '85,15', '90/10', '95/15'})
xticks([1 3 5 7 9 11 13 15 17])
xticklabels({'10/90', '20/80', '30/70', '40/60', '50/50', '60/40', '70/30', '80/20', '90/10'})


load('SVM_28_RMSE_LOG.mat')

new_column = ones(18, 1);
new_row = ones(1, 22);

disp(size(RMSE_log))
for i = 1:18
    cell_avg = 0;
    cell_count = 0;
    for j = 1:22
        if RMSE_log(i,j) < 50
            cell_avg = cell_avg + RMSE_log(i,j);
            cell_count = cell_count + 1;
        end
    end
    new_column(i) = cell_avg/cell_count;
end

hold on
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18];

avgs = new_column';
disp(size(x))
disp(size(avgs))
std_dev = zeros(1,18);

meta_stat = [25, 27, 29, 31, 33, 35, 37, 39];

for i = 1:18
    considered_array = zeros(22,1);
    for j = 1:22
        if RMSE_log(i,j) < 50
            considered_array(j) = RMSE_log(i,j);
        end
    end
    
    std_dev(1,i) = std(nonzeros(considered_array));
end
% plot(x,avgs, 'color', [181/256 102/256 255/256],'LineWidth',1.5)
% patch([x fliplr(x)], [avgs - std_dev fliplr(avgs + std_dev)], [181/256 102/256 255/256], 'LineStyle', 'none');
xlabel('Training/Test Dataset Size', 'Fontsize', 14)
ylabel('RMSE', 'Fontsize', 14)
xticklabels({'10/90','15/85', '20/80', '25/75', '30/70', '35/65', '40/60', '45/55', '50/50', '55/45', '60/40', '65/35', '70/30', '75/25', '80/20', '85,15', '90/10', '95/15'})
xticks([1 3 5 7 9 11 13 15 17])
xticklabels({'10/90', '20/80', '30/70', '40/60', '50/50', '60/40', '70/30', '80/20', '90/10'})
alpha(0.3)
title('SVM-4 RMSE across varying sizes of training and test data', 'Fontsize', 18)
legend('4 SVM version average', '4 SVM version standard deviation', '28 SVM version average', '28 SVM version standard deviation')

