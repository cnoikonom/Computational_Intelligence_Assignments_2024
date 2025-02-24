% Close previous figure tabs, clear the command window, clear workspace variables and clear 'fis' variable
clc;
clear all;
clear fis;
close all;

% Load the Dataset
D = importdata("haberman.data");

%% Split the Data into Training, Evaluation, and Test Sets

% Separate the dataset into two classes based on the last column:
% D1 contains samples where the class label is 1
% D2 contains samples where the class label is 2
D1 = D(D(:, end) == 1, :);
D2 = D(D(:, end) == 2, :);

% Split each class-specific dataset into training, evaluation, and test sets
% using the split_scale function with a scaling parameter of 1
[Dtrn_1, Dval_1, Dchk_1] = split_scale(D1, 1);
[Dtrn_2, Dval_2, Dchk_2] = split_scale(D2, 1);

% Combine the training sets from both classes into a single training set
Dtrn = [Dtrn_1; Dtrn_2];

% Combine the evaluation sets from both classes into a single evaluation set
Dval = [Dval_1; Dval_2];

% Combine the test sets from both classes into a single test set
Dchk = [Dchk_1; Dchk_2];

% Display the sizes of each set
disp(['Training set size: ', num2str(size(Dtrn, 1))]);
disp(['Evaluation set size: ', num2str(size(Dval, 1))]);
disp(['Test set size: ', num2str(size(Dchk, 1))]);

% Shuffle the training data to ensure randomness
nRows = size(Dtrn, 1);
newOrder = randperm(nRows);
temp = Dtrn(newOrder, :);
Dtrn = temp;

% Separate input features and target labels for training, evaluation, and test sets
X_trn = Dtrn(:, 1:end-1);
Y_trn = Dtrn(:, end);
X_val = Dval(:, 1:end-1);
Y_val = Dval(:, end);
X_chk = Dchk(:, 1:end-1);
Y_chk = Dchk(:, end);

%% First ANFIS Model

% Define options for generating the FIS using Subtractive Clustering with a Cluster Influence Range of 0.3
OptionsFis1 = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', 0.3);

% Generate the initial FIS model based on training data and defined options
Fis1 = genfis(X_trn, Y_trn, OptionsFis1);

% Convert output membership functions to constant type for TSK FIS
for i = 1:length(Fis1.output.mf)
    Fis1.output.mf(i).type = 'constant';
    Fis1.output.mf(i).params = Fis1.output.mf(i).params(end); 
end

% Define ANFIS training options with the initial FIS, validation data, and 100 training epochs
options1 = anfisOptions('InitialFis', Fis1, 'ValidationData', Dval, ...
                        'EpochNumber', 100);

% Train the ANFIS model using the training data and defined options
[trnFis1, trnError1, ~, valFis1, valError1] = anfis(Dtrn, options1);

% Predict the class labels for the test set using the trained FIS
Y_pred1 = round(evalfis(X_chk, valFis1));

% Identify unique class labels from the dataset
classes1 = unique(D(:, end));
dim1 = length(classes1);

% Initialize the error matrix for Model 1
errorMatrix1 = zeros(dim1);
N1 = length(Dchk);

% Populate the error matrix by comparing predicted and actual labels
for i = 1:N1
    xpos_1 = find(classes1 == Y_pred1(i));
    ypos_1 = find(classes1 == Y_chk(i));
    errorMatrix1(xpos_1, ypos_1) = errorMatrix1(xpos_1, ypos_1) + 1;
end

% Calculate Overall Accuracy (OA) for Model 1
OA1 = trace(errorMatrix1) / N1;

% Calculate the sum of each row and column in the error matrix
RowsSum1 = sum(errorMatrix1, 2);
ColsSum1 = sum(errorMatrix1, 1);

% Initialize Producer's Accuracy (PA) and User's Accuracy (UA) for each class
PA1 = zeros(1, dim1);
UA1 = zeros(1, dim1);

% Calculate Producer's Accuracy and User's Accuracy for each class
for i = 1:dim1
    PA1(i) = errorMatrix1(i, i) / ColsSum1(i);
    UA1(i) = errorMatrix1(i, i) / RowsSum1(i);
end 

% Calculate K statistic for Model 1
k1 = (N1^2 * OA1 - PA1 .* UA1) / (N1^2 - PA1 .* UA1); 

%% Plotting for First ANFIS Model

% Plot Membership Functions before and after training for each input feature

% Before Training
figure;
for i = 1:length(Fis1.input)
    [xmf, ymf] = plotmf(Fis1, 'input', i);
    subplot(1,length(Fis1.input),i);
    plot(xmf, ymf);
    xlabel('Input');
    ylabel('Degree of Membership');
    title(['Input ' num2str(i)]);
end
sgtitle('Membership Functions Before Training of Model 1');

% After Training
figure;
for i = 1:length(trnFis1.input)
    [xmf, ymf] = plotmf(trnFis1, 'input', i);
    subplot(1, length(Fis1.input), i);
    plot(xmf, ymf);
    xlabel('Input');
    ylabel('Degree of Membership');
    title(['Input ' num2str(i)]);
end
sgtitle('Membership Functions After Training of Model 1');

% Plot Learning Curves (Training and Validation Errors) for Model 1
figure;
plot(trnError1, 'b', 'LineWidth', 1.5);
hold on;
plot(valError1, 'r', 'LineWidth', 1.5);
title('Learning Curves - TSK Model 1'); 
xlabel('Iterations'); 
ylabel('Error');
legend('Training Error', 'Validation Error');
hold off;

%% Second ANFIS Model

% Define options for generating the FIS using Subtractive Clustering with a Cluster Influence Range of 0.8
OptionsFis2 = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', 0.8);

% Generate the initial FIS model based on training data and defined options
Fis2 = genfis(X_trn, Y_trn, OptionsFis2);

% Convert output membership functions to constant type for TSK FIS
for i = 1:length(Fis2.output.mf)
    Fis2.output.mf(i).type = 'constant';
    Fis2.output.mf(i).params = Fis2.output.mf(i).params(end); 
end

% Define ANFIS training options with the initial FIS, validation data, and 100 training epochs
options2 = anfisOptions('InitialFis', Fis2, 'ValidationData', Dval, ...
                        'EpochNumber', 100);

% Train the ANFIS model using the training data and defined options
[trnFis2, trnError2, ~, valFis2, valError2] = anfis(Dtrn, options2);

% Predict the class labels for the test set using the trained FIS
Y_pred2 = round(evalfis(X_chk, valFis2));

% Identify unique class labels from the dataset
classes2 = unique(D(:, end));
dim2 = length(classes2);

% Initialize the error matrix for Model 2
errorMatrix2 = zeros(dim2);
N2 = length(Dchk);

% Populate the error matrix by comparing predicted and actual labels
for i = 1:N2
    xpos_2 = find(classes2 == Y_pred2(i));
    ypos_2 = find(classes2 == Y_chk(i));
    errorMatrix2(xpos_2, ypos_2) = errorMatrix2(xpos_2, ypos_2) + 1;
end

% Calculate Overall Accuracy (OA) for Model 2
OA2 = trace(errorMatrix2) / N2;

% Calculate the sum of each row and column in the error matrix
RowsSum2 = sum(errorMatrix2, 2);
ColsSum2 = sum(errorMatrix2, 1);

% Initialize Producer's Accuracy (PA) and User's Accuracy (UA) for each class
PA2 = zeros(1, dim2);
UA2 = zeros(1, dim2);

% Calculate Producer's Accuracy and User's Accuracy for each class
for i = 1:dim2
    PA2(i) = errorMatrix2(i, i) / ColsSum2(i);
    UA2(i) = errorMatrix2(i, i) / RowsSum2(i);
end 

% Calculate K statistic for Model 2
k2 = (N2^2 * OA2 - PA2 .* UA2) / (N2^2 - PA2 .* UA2); 

%% Plotting for Second ANFIS Model

% Plot Membership Functions before and after training for each input feature

% Before Training
figure;
for i = 1:length(Fis2.input)
    [xmf, ymf] = plotmf(Fis2, 'input', i);
    subplot(1, length(Fis2.input), i);
    plot(xmf, ymf);
    xlabel('Input');
    ylabel('Degree of Membership');
    title(['Input ' num2str(i)]);
end
sgtitle('Membership Functions Before Training of Model 2');

% After Training
figure;
for i = 1:length(trnFis2.input)
    [xmf, ymf] = plotmf(trnFis2, 'input', i);
    subplot(1, length(trnFis2.input), i);
    plot(xmf, ymf);
    xlabel('Input');
    ylabel('Degree of Membership');
    title(['Input ' num2str(i)]);
end
sgtitle('Membership Functions After Training of Model 2');

% Plot Learning Curves (Training and Validation Errors) for Model 2
figure;
plot(trnError2, 'b', 'LineWidth', 1.5);
hold on;
plot(valError2, 'r', 'LineWidth', 1.5);
title('Learning Curves - TSK Model 2'); 
xlabel('Iterations'); 
ylabel('Error');
legend('Training Error', 'Validation Error');
hold off;

%% Third ANFIS Model

% Note: Since there are two classes, perform Subtractive Clustering separately for each class

% Perform Subtractive Clustering on Class 1 with a Cluster Influence Range of 0.3
[c1_3, sig1_3] = subclust(Dtrn(Y_trn == 1, :), 0.3);

% Perform Subtractive Clustering on Class 2 with a Cluster Influence Range of 0.3
[c2_3, sig2_3] = subclust(Dtrn(Y_trn == 2, :), 0.3);

% Total number of rules is the sum of clusters from both classes
num_rules_3 = size(c1_3, 1) + size(c2_3, 1);

% Initialize a new Sugeno-type FIS
fis3 = newfis('FIS_SC', 'sugeno');

% Define names for input variables
names_in = {'in1', 'in2', 'in3'};

% Add input variables to the FIS with a range of [0, 1]
for i = 1:size(Dtrn, 2) - 1
    fis3 = addvar(fis3, 'input', names_in{i}, [0 1]);
end

% Add output variable to the FIS with a range of [1, 2]
fis3 = addvar(fis3, 'output', 'out1', [1 2]);

% Add Input Membership Functions using Gaussian MFs based on Subtractive Clustering
for i = 1:size(Dtrn, 2) - 1
    % Add MFs for Class 1
    for j = 1:size(c1_3, 1)
        name = ['sth' num2str(i) '_c1_3' num2str(j)];
        fis3 = addmf(fis3, 'input', i, name, 'gaussmf', [sig1_3(i) c1_3(j, i)]); 
    end
    % Add MFs for Class 2
    for j = 1:size(c2_3, 1)
        name = ['sth' num2str(i) '_c2_3' num2str(j)];
        fis3 = addmf(fis3, 'input', i, name, 'gaussmf', [sig2_3(i) c2_3(j, i)]);
    end
end

% Define output parameters for each rule (constant values corresponding to classes)
params_3 = [ones(1, size(c1_3, 1)), 2 * ones(1, size(c2_3, 1))];  

% Add Output Membership Functions as constant values
for i = 1:num_rules_3
    name = ['sth' num2str(i) 'params_3' num2str(j)];
    fis3 = addmf(fis3, 'output', 1, name, 'constant', params_3(i));
end

% Initialize the rule list matrix with zeros
% Each row corresponds to a rule and each column to an input/output
ruleList_3 = zeros(num_rules_3, size(Dtrn, 2));

% Assign rule indices to the rule list
for i = 1:size(ruleList_3, 1)
    ruleList_3(i, :) = i;
end

% Append the output class labels to the rule list
ruleList_3 = [ruleList_3, ones(num_rules_3, 2)];

% Add the rules to the FIS
fis3 = addrule(fis3, ruleList_3);

% Define ANFIS training options with the initial FIS, validation data, and 100 training epochs
options = anfisOptions('InitialFis', fis3, 'ValidationData', Dval, ...
                       'EpochNumber', 100);

% Train the ANFIS model using the training data and defined options
[trnFis3, trnError3, ~, valFis3, valError3] = anfis(Dtrn, options);

% Predict the class labels for the test set using the trained FIS
Y_pred3 = round(evalfis(X_chk, valFis3));

% Identify unique class labels from the dataset
classes3 = unique(D(:, end));
dim3 = length(classes3);

% Initialize the error matrix for Model 3
errorMatrix3 = zeros(dim3);
N3 = length(Dchk);

% Populate the error matrix by comparing predicted and actual labels
for i = 1:N3
    xpos_3 = find(classes3 == Y_pred3(i));
    ypos_3 = find(classes3 == Y_chk(i));
    errorMatrix3(xpos_3, ypos_3) = errorMatrix3(xpos_3, ypos_3) + 1;
end

% Calculate Overall Accuracy (OA) for Model 3
OA3 = trace(errorMatrix3) / N3;

% Calculate the sum of each row and column in the error matrix
RowsSum3 = sum(errorMatrix3, 2);
ColsSum3 = sum(errorMatrix3, 1);

% Initialize Producer's Accuracy (PA) and User's Accuracy (UA) for each class
PA3 = zeros(1, dim3);
UA3 = zeros(1, dim3);

% Calculate Producer's Accuracy and User's Accuracy for each class
for i = 1:dim3
    PA3(i) = errorMatrix3(i, i) / ColsSum3(i);
    UA3(i) = errorMatrix3(i, i) / RowsSum3(i);
end 

% Calculate K statistic for Model 3
k3 = (N3^2 * OA3 - PA3 .* UA3) / (N3^2 - PA3 .* UA3); 

%% Plotting for Third ANFIS Model

% Plot Membership Functions before and after training for each input feature

% Before Training
figure;
for i = 1:length(fis3.input)
    [xmf, ymf] = plotmf(fis3, 'input', i);
    subplot(1, length(fis3.input), i);
    plot(xmf, ymf);
    xlabel('Input');
    ylabel('Degree of Membership');
    title(['Input ' num2str(i)]);
end
sgtitle('Membership Functions Before Training of Model 3');

% After Training
figure;
for i = 1:length(trnFis3.input)
    [xmf, ymf] = plotmf(trnFis3, 'input', i);
    subplot(1, length(trnFis3.input), i);
    plot(xmf, ymf);
    xlabel('Input');
    ylabel('Degree of Membership');
    title(['Input ' num2str(i)]);
end
sgtitle('Membership Functions After Training of Model 3');

% Plot Learning Curves (Training and Validation Errors) for Model 3
figure;
plot(trnError3, 'b', 'LineWidth', 1.5);
hold on;
plot(valError3, 'r', 'LineWidth', 1.5);
title('Learning Curves - TSK Model 3'); 
xlabel('Iterations'); 
ylabel('Error');
legend('Training Error', 'Validation Error');
hold off;

%% Fourth ANFIS Model

% Note: Since there are two classes, perform Subtractive Clustering separately for each class

% Perform Subtractive Clustering on Class 1 with a Cluster Influence Range of 0.8
[c1_4, sig1_4] = subclust(Dtrn(Y_trn == 1, :), 0.8);

% Perform Subtractive Clustering on Class 2 with a Cluster Influence Range of 0.8
[c2_4, sig2_4] = subclust(Dtrn(Y_trn == 2, :), 0.8);

% Total number of rules is the sum of clusters from both classes
num_rules_4 = size(c1_4, 1) + size(c2_4, 1);

% Initialize a new Sugeno-type FIS
fis4 = newfis('FIS_SC', 'sugeno');

% Define names for input variables
names_in = {'in1', 'in2', 'in3'};

% Add input variables to the FIS with a range of [0, 1]
for i = 1:size(Dtrn, 2) - 1
    fis4 = addvar(fis4, 'input', names_in{i}, [0 1]);
end

% Add output variable to the FIS with a range of [1, 2]
fis4 = addvar(fis4, 'output', 'out1', [1 2]);

% Add Input Membership Functions using Gaussian MFs based on Subtractive Clustering
for i = 1:size(Dtrn, 2) - 1
    % Add MFs for Class 1
    for j = 1:size(c1_4, 1)
        name = ['sth' num2str(i) '_c1_4' num2str(j)];
        fis4 = addmf(fis4, 'input', i, name, 'gaussmf', [sig1_4(i) c1_4(j, i)]); 
    end
    % Add MFs for Class 2
    for j = 1:size(c2_4, 1)
        name = ['sth' num2str(i) '_c2_4' num2str(j)];
        fis4 = addmf(fis4, 'input', i, name, 'gaussmf', [sig2_4(i) c2_4(j, i)]);
    end
end

% Define output parameters for each rule (constant values corresponding to classes)
params_4 = [ones(1, size(c1_4, 1)), 2 * ones(1, size(c2_4, 1))];  

% Add Output Membership Functions as constant values
for i = 1:num_rules_4
    name = ['sth' num2str(i) 'params_4' num2str(j)];
    fis4 = addmf(fis4, 'output', 1, name, 'constant', params_4(i));
end

% Initialize the rule list matrix with zeros
% Each row corresponds to a rule and each column to an input/output
ruleList_4 = zeros(num_rules_4, size(Dtrn, 2));

% Assign rule indices to the rule list
for i = 1:size(ruleList_4, 1)
    ruleList_4(i, :) = i;
end

% Append the output class labels to the rule list
ruleList_4 = [ruleList_4, ones(num_rules_4, 2)];

% Add the rules to the FIS
fis4 = addrule(fis4, ruleList_4);

% Define ANFIS training options with the initial FIS, validation data, and 100 training epochs
options = anfisOptions('InitialFis', fis4, 'ValidationData', Dval, ...
                       'EpochNumber', 100);

% Train the ANFIS model using the training data and defined options
[trnFis4, trnError4, ~, valFis4, valError4] = anfis(Dtrn, options);

% Predict the class labels for the test set using the trained FIS
Y_pred4 = round(evalfis(X_chk, valFis4));

% Identify unique class labels from the dataset
classes4 = unique(D(:, end));
dim4 = length(classes4);

% Initialize the error matrix for Model 4
errorMatrix4 = zeros(dim4);
N4 = length(Dchk);

% Populate the error matrix by comparing predicted and actual labels
for i = 1:N4
    xpos_4 = find(classes4 == Y_pred4(i));
    ypos_4 = find(classes4 == Y_chk(i));
    errorMatrix4(xpos_4, ypos_4) = errorMatrix4(xpos_4, ypos_4) + 1;
end

% Calculate Overall Accuracy (OA) for Model 4
OA4 = trace(errorMatrix4) / N4;

% Calculate the sum of each row and column in the error matrix
RowsSum4 = sum(errorMatrix4, 2);
ColsSum4 = sum(errorMatrix4, 1);

% Initialize Producer's Accuracy (PA) and User's Accuracy (UA) for each class
PA4 = zeros(1, dim4);
UA4 = zeros(1, dim4);

% Calculate Producer's Accuracy and User's Accuracy for each class
for i = 1:dim4
    PA4(i) = errorMatrix4(i, i) / ColsSum4(i);
    UA4(i) = errorMatrix4(i, i) / RowsSum4(i);
end 

% Calculate K statistic for Model 4
k4 = (N4^2 * OA4 - PA4 .* UA4) / (N4^2 - PA4 .* UA4); 

% Compile performance metrics for all models into a matrix
% Each row corresponds to a model and columns to [OA, PA, UA, K]
performance_metrics = [OA1, PA1, UA1, k1; 
             OA2, PA2, UA2, k2; 
             OA3, PA3, UA3, k3; 
             OA4, PA4, UA4, k4];

%% Plotting for Fourth ANFIS Model

% Plot Membership Functions before and after training for each input feature

% Before Training
figure;
for i = 1:length(fis4.input)
    [xmf, ymf] = plotmf(fis4, 'input', i);
    subplot(1, length(fis4.input), i);
    plot(xmf, ymf);
    xlabel('Input');
    ylabel('Degree of Membership');
    title(['Input ' num2str(i)]);
end
sgtitle('Membership Functions Before Training of Model 4');

% After Training
figure;
for i = 1:length(trnFis4.input)
    [xmf, ymf] = plotmf(trnFis4, 'input', i);
    subplot(1, length(trnFis4.input), i);
    plot(xmf, ymf);
    xlabel('Input');
    ylabel('Degree of Membership');
    title(['Input ' num2str(i)]);
end
sgtitle('Membership Functions After Training of Model 4');

% Plot Learning Curves (Training and Validation Errors) for Model 4
figure;
plot(trnError4, 'b', 'LineWidth', 1.5);
hold on;
plot(valError4, 'r', 'LineWidth', 1.5);
title('Learning Curves - TSK Model 4'); 
xlabel('Iterations'); 
ylabel('Error');
legend('Training Error', 'Validation Error');
hold off;

%% Display Error Matrices and Model Performance Metrics

% Display the error matrices for all four models
disp('Error Matrix for Model 1:');
disp(errorMatrix1);
disp('Error Matrix for Model 2:');
disp(errorMatrix2);
disp('Error Matrix for Model 3:');
disp(errorMatrix3);
disp('Error Matrix for Model 4:');
disp(errorMatrix4);

% Display the overall performance metrics for all models
disp('Performance Metrics for All Models:');
disp('Columns: [OA, PA1, PA2, UA1, UA2, K]');
disp(performance_metrics);
