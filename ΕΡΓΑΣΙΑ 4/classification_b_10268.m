% Close all open figures, clear all variables, and clear the command window
close all;
clear all;
clc;

% Load the dataset from 'epileptic_seizure_data.csv' and extract the data
D = importdata('epileptic_seizure_data.csv');
D = D.data;

%% Split data based on class labels (1-5)
D1 = D(D(:,end) == 1, :);
D2 = D(D(:,end) == 2, :);
D3 = D(D(:,end) == 3, :);
D4 = D(D(:,end) == 4, :);
D5 = D(D(:,end) == 5, :);

% Perform scaling and split each class into training, validation, and test sets
[Dtrn_1,Dval_1,Dchk_1] = split_scale(D1,1);
[Dtrn_2,Dval_2,Dchk_2] = split_scale(D2,1);
[Dtrn_3,Dval_3,Dchk_3] = split_scale(D3,1);
[Dtrn_4,Dval_4,Dchk_4] = split_scale(D4,1);
[Dtrn_5,Dchk_5,Dval_5] = split_scale(D5,1);

% Combine the training, validation, and test data from all classes
Dtrn = [Dtrn_1 ; Dtrn_2 ; Dtrn_3 ; Dtrn_4 ; Dtrn_5];
Dval = [Dval_1 ; Dval_2 ; Dval_3 ; Dval_4 ; Dval_5];
Dchk = [Dchk_1 ; Dchk_2 ; Dchk_3 ; Dchk_4 ; Dchk_5];

% Display the sizes of each set
disp(['Training set size: ', num2str(size(Dtrn, 1))]);
disp(['Evaluation set size: ', num2str(size(Dval, 1))]);
disp(['Test set size: ', num2str(size(Dchk, 1))]);

% Shuffle the training data
nRows = size(Dtrn, 1);
newOrder = randperm(nRows);
temp = Dtrn(newOrder, :);
Dtrn = temp;

% Separate features and labels for training, validation, and test data
X_train = Dtrn(:,1:end-1);
Y_train = Dtrn(:,end);
X_val = Dval(:,1:end-1);
Y_val = Dval(:,end);
X_chk = Dchk(:,1:end-1);
Y_chk = Dchk(:,end);

% Feature selection using the ReliefF algorithm, selecting 10 best features
[idx,~] = relieff(X_train, Y_train,10);

featuresNum = [4 8 12 14];
R = [0.3 0.6 0.9];

%% 5-fold Validation Setup
error = zeros(length(featuresNum),length(R)); % Initialize error matrix

%% Grid Search & 5 Fold Cross Validation
for nf = 1:length(featuresNum) % Iterate through feature numbers
    for r = 1:length(R) % Iterate through radius values
        cv = cvpartition(length(Dtrn(:,1)), 'KFold', 5); % 5-fold partition
        for k=1:5   % Loop over the folds
            % Select features based on ReliefF ranking        
            selected_features = [Dtrn(:,idx(1:featuresNum(nf))) Y_train];

            % Get training and validation indices for current fold
            train_idx = training(cv,k);
            val_idx = test(cv,k); 
            
            % Get training and validation data based on indices
            vali = test(cv,k); 
            train = training(cv,k);
            
            valIdx = find(vali == 1);
            trainIdx = find(train == 1);
          
            training_data_new = selected_features(trainIdx,:);
            validation_data_new = selected_features(valIdx,:);

            % Perform subtractive clustering for each class in training data 
            [c1,sig1] = subclust(training_data_new(training_data_new(:,end)== 1,:),R(r));
            [c2,sig2] = subclust(training_data_new(training_data_new(:,end)== 2,:),R(r));
            [c3,sig3] = subclust(training_data_new(training_data_new(:,end)== 3,:),R(r));
            [c4,sig4] = subclust(training_data_new(training_data_new(:,end)== 4,:),R(r));
            [c5,sig5] = subclust(training_data_new(training_data_new(:,end)== 5,:),R(r));
            
            num_rules = size(c1,1) + size(c2,1) + size(c3,1) + size(c4,1) + size(c5,1);
            
            % Create a new FIS (fuzzy inference system) of Sugeno type
            fis = newfis('FIS_SC','sugeno');

            for i = 1:size(training_data_new,2)-1
                % Define the name for each input variable
                names_in{i} = sprintf("Input %d",i);
            end

            for i=1:size(training_data_new,2)-1
                % Add the input variable with a range [0, 1]
                fis = addvar(fis,'input',names_in{i},[0 1]);
            end


            % Add output variable to the FIS
            % The output range is [1, 5] corresponding to the class labels
            fis = addvar(fis,'output','Output 1',[1 5]);
        
            % Add Input Membership Functions

            for i=1:size(training_data_new,2)-1
                for j=1:size(c1,1)
                    name = ['sth' num2str(i) '_c1' num2str(j)];
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]);
                end
                for j=1:size(c2,1)
                    name = ['sth' num2str(i) '_c2' num2str(j)];
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
                end
                for j=1:size(c3,1)
                    name = ['sth' num2str(i) '_c3' num2str(j)];
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig3(i) c3(j,i)]);
                end
                for j=1:size(c4,1)
                    name = ['sth' num2str(i) '_c4' num2str(j)];
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig4(i) c4(j,i)]);
                end
                for j=1:size(c5,1)
                    name = ['sth' num2str(i) '_c5' num2str(j)];
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig5(i) c5(j,i)]);
                end
            end
            params = [ones(1,size(c1,1)) 2*ones(1,size(c2,1)) 3*ones(1,size(c3,1)) ...
                    4*ones(1,size(c4,1)) 5*ones(1,size(c5,1))];
            
            % Add Output Membership Functions
            for i = 1:num_rules
                name = ['sth' num2str(i) 'params' num2str(j)];
                fis = addmf(fis,'output',1,name,'constant',params(i));
            end

            % Add FIS Rule Base
            ruleList = zeros(num_rules,size(training_data_new,2));
            disp(size(ruleList,1));
            for i=1:size(ruleList,1)
                ruleList(i,:)=i;
            end
            ruleList = [ruleList ones(num_rules,2)];
            fis = addrule(fis,ruleList);
            [trnFis,trnError,~,valFis,valError] = anfis...
                (training_data_new,fis,[100 0 0.01 0.9 1.1],[],validation_data_new);
            error (nf,r) = error(nf,r) + mean(valError);

        end
    end
end

error = error/5;

%% Find Optimal Hyperparameters
min_error = min(min(error));
[optimal_featuresNum, optimal_NR] = find(error == min_error);

train = [Dtrn(:,idx(1:featuresNum(optimal_featuresNum))) Y_train];
validation = [Dval(:,idx(1:featuresNum(optimal_featuresNum))) Y_val];
check = [Dchk(:,idx(1:featuresNum(optimal_featuresNum))) Y_chk];
optimalR = R(optimal_NR);

%% Final Model

[c1,sig1] = subclust(train(train(:,end)==1,:),optimalR);
[c2,sig2] = subclust(train(train(:,end)==2,:),optimalR);
[c3,sig3] = subclust(train(train(:,end)==3,:),optimalR);
[c4,sig4] = subclust(train(train(:,end)==4,:),optimalR);
[c5,sig5] = subclust(train(train(:,end)==5,:),optimalR);

num_rules = size(c1,1)+size(c2,1)+size(c3,1)+size(c4,1)+size(c5,1);

fis = newfis('FIS_SC','sugeno');

for i = 1:size(train,2)-1
      names_in{i} = sprintf("sth%d",i);
end
for i = 1:size(train,2)-1
    fis = addvar(fis,'input',names_in{i},[0 1]);
end
fis = addvar(fis,'output','out1',[1 5]); 

for i = 1:size(train,2)-1
    for j = 1:size(c1,1)
        name = ['sth' num2str(i) '_c1' num2str(j)];
        fis = addmf(fis,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]); 
    end
    for j = 1:size(c2,1)
        name = ['sth' num2str(i) '_c2' num2str(j)];
        fis = addmf(fis,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
    end
    for j = 1:size(c3,1)
        name = ['sth' num2str(i) '_c3' num2str(j)];
        fis = addmf(fis,'input',i,name,'gaussmf',[sig3(i) c3(j,i)]);
    end
    for j = 1:size(c4,1)
        name = ['sth' num2str(i) '_c4' num2str(j)];
        fis = addmf(fis,'input',i,name,'gaussmf',[sig4(i) c4(j,i)]);
    end
    for j = 1:size(c5,1)
        name = ['sth' num2str(i) '_c5' num2str(j)];
        fis = addmf(fis,'input',i,name,'gaussmf',[sig5(i) c5(j,i)]);    
    end
end

params = [ones(1,size(c1,1)) 2*ones(1,size(c2,1)) 3*ones(1,size(c3,1)) 4*ones(1,size(c4,1)) 5*ones(1,size(c5,1))];  
for i = 1:num_rules
    name = ['sth' num2str(i) 'params' num2str(j)];
    fis = addmf(fis,'output',1,name,'constant',params(i));
end

%Add FIS Rule Base
ruleList=zeros(num_rules,size(train,2));
for i=1:size(ruleList,1)
    ruleList(i,:) = i;
end
ruleList = [ruleList ones(num_rules,2)];
fis = addrule(fis,ruleList);

% Train & Evaluate AfeaturesNumIS
[trnFis,trnError,~,valFis,valError] = anfis(train,fis,[100 0 0.01 0.9 1.1],[],validation);
Y_pred = round(evalfis(check(:,1:end-1),valFis));

classes = unique(D(:,end));
dim = length(classes);
errorMatrix = zeros(dim);
N = length(check);

for i = 1:N
    xpos = find(classes == Y_pred(i));
    ypos = find(classes == check(i, end));
    errorMatrix(xpos, ypos) = errorMatrix(xpos, ypos) + 1;
end

% Overall Accuracy
OA = trace(errorMatrix) / N;
RowsSum = sum(errorMatrix,2);
ColsSum = sum(errorMatrix,1);
PA = zeros(1, dim);
UA = zeros(1, dim);
for i = 1:dim
    PA(i) = errorMatrix(i,i) / ColsSum(i);
    UA(i) = errorMatrix(i,i) / RowsSum(i);
end 

k = (N^2 * OA - PA .* UA) / (N^2 - PA .* UA); 

errorMatrix;
performance_metrics = [OA,PA,UA,k];

%% Plots

% Mean Error Respective to Number of Features
figure
hold on
title('Mean Error - Number of Features');
xlabel('Number of Features')
ylabel('Mean Error')
plot(featuresNum, error(:, 1));
plot(featuresNum, error(:, 2));
plot(featuresNum, error(:, 3));
legend('r = 0.3', 'r = 0.6', 'r = 0.9');

% Mean error respective to the cluster radius
figure
hold on
title('Mean Error - Radius');
xlabel('Radius')
ylabel('Error')
plot(R, error(1, :));
plot(R, error(2, :));
plot(R, error(3, :));
plot(R, error(4, :));
legend('4 Features', '8 Features', '12 Features','14 Features');

% Predictions - Real Values

figure;
hold on
title('Predictions - Real Values');
xlabel('Test Dataset')
ylabel('y_pred - y_real')
scatter(1:length(Y_pred), Y_pred);
scatter(1:length(Y_pred), check(:, end));
legend('Predictions', 'Real Values');

% Learning Curves

figure;
plot([trnError valError]);
legend('Training Error','Validation Error');
xlabel('Iterations');
ylabel('Error');

% Plot of Membership Functions
if (size(train,2)-1) == 4
figure;
    for i = 1:(size(train,2)-1)
        subplot(4,1,i)
        plotmf(fis, 'input',i);
    end
elseif (size(train,2)-1) == 8 
    figure;
    for i = 1:(size(train,2)-1)/2
        subplot(4,1,i)
        plotmf(fis, 'input',i);
    end
    figure;
    for i = 1:(size(train,2)-1)/2
        subplot(4,1,i)
        plotmf(fis, 'input',i+4);    
    end
elseif (size(train,2)-1) == 12
    figure;
    for i = 1:(size(train,2)-1)/3
        subplot(4,1,i)
        plotmf(fis, 'input',i);
    end
    figure;
    for i = 1:(size(train,2)-1)/3
        subplot(4,1,i)
        plotmf(fis, 'input',i+4);    
    end
    figure;
    for i = 1:(size(train,2)-1)/3
        subplot(4,1,i)
        plotmf(fis, 'input',i+8);    
    end
else 
figure;
    for i = 1:4
        subplot(4,1,i)
        plotmf(fis, 'input',i);
    end
    figure;
    for i = 1:4
        subplot(4,1,i)
        plotmf(fis, 'input',i+4);    
    end
    figure;
    for i = 1:4
        subplot(4,1,i)
        plotmf(fis, 'input',i+8);    
    end
    figure;
    for i = 1:2
        subplot(2,1,i)
        plotmf(fis, 'input',i+12);    
    end
end



if (size(train,2)-1) == 4
figure;
    for i = 1:(size(train,2)-1)
        subplot(4,1,i)
        plotmf(trnFis, 'input',i);
    end
elseif (size(train,2)-1) == 8 
    figure;
    for i = 1:(size(train,2)-1)/2
        subplot(4,1,i)
        plotmf(trnFis, 'input',i);
    end
    figure;
    for i = 1:(size(train,2)-1)/2
        subplot(4,1,i)
        plotmf(trnFis, 'input',i+4);    
    end
elseif (size(train,2)-1) == 12
    figure;
    for i = 1:(size(train,2)-1)/3
        subplot(4,1,i)
        plotmf(trnFis, 'input',i);
    end
    figure;
    for i = 1:(size(train,2)-1)/3
        subplot(4,1,i)
        plotmf(trnFis, 'input',i+4);    
    end
    figure;
    for i = 1:(size(train,2)-1)/3
        subplot(4,1,i)
        plotmf(trnFis, 'input',i+8);    
    end
else 
    figure;
    for i = 1:4
        subplot(4,1,i)
        plotmf(trnFis, 'input',i);
    end
    figure;
    for i = 1:4
        subplot(4,1,i)
        plotmf(trnFis, 'input',i+4);    
    end
    figure;
    for i = 1:4
        subplot(4,1,i)
        plotmf(trnFis, 'input',i+8);    
    end
    figure;
    for i = 1:2
        subplot(2,1,i)
        plotmf(trnFis, 'input',i+12);    
    end
end