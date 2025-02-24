% Close previous figure tabs and clear the command window
close all;
clear all;
clc;

% Load the dataset
D = importdata('superconduct.csv');

%% Split the data into three partitions 
% Dtrn --> Training Set Data (60%)
% Dval --> Evaluation Set Data (20%)
% Dchk --> Test Set Data (20%)
[Dtrn,Dval,Dchk] = split_scale(D,1);

% Display the sizes of each set
disp(['Training set size: ', num2str(size(Dtrn, 1))]);
disp(['Evaluation set size: ', num2str(size(Dval, 1))]);
disp(['Test set size: ', num2str(size(Dchk, 1))]);


%% Set Grid Search Parameters

% Define the range of values for the number of features to select
feature_values = [4, 8, 12, 16];

% Range of values for the cluster influence radius used in Subtractive Clustering
r = [0.3, 0.5, 0.8, 1]; 

% Number of folds for k-fold cross-validation
k = 5;

% Initialize matrices to store cross-validation errors and number of rules
errors = zeros(length(feature_values), length(r));
rules = zeros(length(r), 1);

%% Perform Grid Search with Cross-Validation

% Iterate over each number of features
for p = 1:length(feature_values)
    NumOfFeatures = feature_values(p);  % Current number of features to select
    
    % Iterate over each cluster influence radius
    for j = 1:length(r)
        rad = r(j);  % Current cluster influence radius
        
        % Create a k-fold cross-validation partition for the training data
        c = cvpartition(size(Dtrn, 1), 'KFold', k);
        
        % Initialize vectors to store cross-validation errors and number of rules for each fold
        cv_errors = zeros(k, 1);
        temp_rules = zeros(k, 1);
        
        % Perform k-fold cross-validation
        for i = 1:k
            %% Get Current Fold's Training and Validation Indices
            
            % Retrieve logical indices for training and validation sets for the current fold
            train_idx = training(c, i);
            val_idx = test(c, i);

            % Extract input features and target values for training
            x_trn = Dtrn(train_idx, 1:end-1);
            y_trn = Dtrn(train_idx, end);

            % Extract input features and target values for validation
            x_val = Dtrn(val_idx, 1:end-1);
            y_val = Dtrn(val_idx, end);

            %% Feature Selection Using ReliefF
            
            % Identify the most relevant features using the ReliefF algorithm
            % '10' specifies the number of nearest neighbors to consider
            [feat_idx, ~] = relieff(x_trn, y_trn, 10);
            
            % Select the top 'NumOfFeatures' features based on ReliefF ranking
            feat_idx = feat_idx(1:NumOfFeatures);
            
            %% Select Selected Features for Training and Validation Sets
            
            % Reduce the training and validation input data to the selected features
            x_trn = x_trn(:, feat_idx);
            x_val = x_val(:, feat_idx);
            
            % Combine the selected features with the target values
            data_trn = [x_trn y_trn];
            data_val = [x_val y_val];
            
            %% Generate Fuzzy Inference System (FIS) using Subtractive Clustering
            
            % Set options for generating the FIS with Subtractive Clustering
            gen_options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', rad);
            
            % Generate the initial Takagi-Sugeno-Kang (TSK) FIS model
            tsk_model = genfis(x_trn, y_trn, gen_options);
            
            %% Train the ANFIS Model
            
            % Set ANFIS training options
            anfis_options = anfisOptions("InitialFis", tsk_model, ...
                                   "ValidationData", data_val, ...
                                   "EpochNumber", 40);
            
            % Train the ANFIS model using the training data and options
            [~, ~, ~, valFIS, valError] = anfis(data_trn, anfis_options);

            %% Record Cross-Validation Error
            
            % Calculate and store the minimum validation error squared for the current fold
            cv_errors(i) = min(valError .* valError);
        end
        
        %% Aggregate Cross-Validation Results
        
        % Compute the mean cross-validation error across all folds for current hyperparameters
        errors(p, j) = mean(cv_errors);
        
        % Record the number of rules from the last fold's best FIS
        rules(j) = size(showrule(valFIS), 1);
    end
end

%% Visualize Grid Search Results

% Plot Mean Square Error (MSE) against the number of rules for each feature count
figure;
for i = 1:length(feature_values)
    subplot(2, 2, i)
    plot(rules, errors(i, :), '-o', 'LineWidth', 2, 'MarkerSize', 6)
    title(sprintf('Number of Features = %d', feature_values(i)))
    xlabel('Number of Rules')
    ylabel('Mean Square Error')
    grid on;
end
sgtitle('MSE vs. Number of Rules for Different Feature Counts');

% Plot Mean Square Error (MSE) against the number of features for each rule count
figure;
for i = 1:length(r)
    subplot(2, 2, i)
    plot(feature_values, errors(:, i), '-s', 'LineWidth', 2, 'MarkerSize', 6)
    title(sprintf('Cluster Influence Radius = %.1f', r(i)))
    xlabel('Number of Features')
    ylabel('Mean Square Error')
    grid on;
end
sgtitle('MSE vs. Number of Features for Different Cluster Radii');

%% Identify Optimal Hyperparameters

% Find the minimum MSE across all grid search results
min_error = min(errors, [], 'all');

% Locate the indices (feature count and radius) corresponding to the minimum MSE
[optimal_feat_idx, optimal_radius_idx] = find(errors == min_error);

% Retrieve the optimal number of features and cluster influence radius
optimal_features = feature_values(optimal_feat_idx);
optimal_radius = r(optimal_radius_idx);

% Display the optimal hyperparameters
fprintf('Optimal Number of Features: %d\n', optimal_features);
fprintf('Optimal Cluster Influence Radius: %.1f\n', optimal_radius);

%% Train Final ANFIS Model with Optimal Hyperparameters

% Extract input features and target values from training, evaluation, and test sets
X_trn = Dtrn(:, 1:end-1);
Y_trn = Dtrn(:, end);
X_val = Dval(:, 1:end-1);
Y_val = Dval(:, end);
X_chk = Dchk(:, 1:end-1);
Y_chk = Dchk(:, end);

% Perform feature selection on the entire training set using ReliefF
[feat_idx, ~] = relieff(X_trn, Y_trn, 10);
feat_idx = feat_idx(1:optimal_features);  % Select top 'optimal_features' features

%% Select Optimal Features for All Data Partitions

% Reduce the training, validation, and test sets to the selected optimal features
X_trn = X_trn(:, feat_idx);
X_val = X_val(:, feat_idx);
X_chk = X_chk(:, feat_idx);

%% Generate FIS with Optimal Hyperparameters

% Set options for generating the FIS with the optimal cluster influence radius
gen_options = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', optimal_radius);

% Generate the initial TSK FIS model using the optimal hyperparameters
tsk_model = genfis(X_trn, Y_trn, gen_options);

%% Train the Final ANFIS Model

% Set ANFIS training options for the final model
anfis_options = anfisOptions("InitialFis", tsk_model, ...
                      "ValidationData", [X_val Y_val], ...
                      "EpochNumber", 40, ...
                      "OptimizationMethod", 1);

% Train the ANFIS model using the training data and options
[fis, trainError, stepSize, valFIS, valError] = anfis([X_trn Y_trn], anfis_options);

%% Evaluate the Final Model on Test Data

% Predict the critical temperatures for the test set using the trained FIS
y_pred = evalfis(valFIS, X_chk);

% Calculate Root Mean Squared Error (RMSE) between actual and predicted values
rmse = sqrt(mse(Y_chk, y_pred));    

% Calculate Coefficient of Determination (R²)
r2 = 1 - sum((y_pred - Y_chk).^2) / sum((Y_chk - mean(Y_chk)).^2);

% Calculate Normalized Mean Squared Error (NMSE)
nmse = 1 - r2;

% Calculate Normalized Root Mean Squared Error (NDEI)
ndei = sqrt(nmse);

% Store the performance metrics
performance_metrics = [rmse nmse ndei r2];

% Display the performance metrics
disp('Performance Metrics for the Final Model:');
disp('Columns: [RMSE, NMSE, NDEI, R²]');
disp(performance_metrics);

%% Visualize Results

% Plot Actual Critical Temperatures
figure;
scatter(1:length(Y_chk), Y_chk, 2, 'b', 'filled')
title("Actual Critical Temperatures")
xlabel("Data Point")
ylabel("Critical Temperature")
grid on;

% Plot Predicted Critical Temperatures
figure;
scatter(1:length(y_pred), y_pred, 2, 'r', 'filled')
title("Predicted Critical Temperatures")
xlabel("Data Point")
ylabel("Critical Temperature")
grid on;

% Plot Training and Validation Errors Over Epochs
figure;
plot(trainError, 'b', 'LineWidth', 2)
hold on
plot(valError, 'r', 'LineWidth', 2)
title("Training and Validation Errors Over Epochs")
xlabel('Epochs')
ylabel('Error')
legend('Training Error', 'Validation Error', 'Location', 'Best')
grid on;
hold off

% Plot Membership Functions Before and After Training for the First Two Features
figure;
subplot(2,2,1)
plotmf(tsk_model, 'input', 1)
xlabel('First Feature')
title('Membership Functions Before Training (Feature 1)')

subplot(2,2,2)
plotmf(valFIS, 'input', 1)
xlabel('First Feature')
title('Membership Functions After Training (Feature 1)')

subplot(2,2,3)
plotmf(tsk_model, 'input', 2)
xlabel('Second Feature')
title('Membership Functions Before Training (Feature 2)')

subplot(2,2,4)
plotmf(valFIS, 'input', 2)
xlabel('Second Feature')
title('Membership Functions After Training (Feature 2)')