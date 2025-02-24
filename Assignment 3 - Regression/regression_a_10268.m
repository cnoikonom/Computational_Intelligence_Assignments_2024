% Close previous figure tabs and clear the command window
close all
clc
clear 

% Load the dataset
D = importdata('airfoil_self_noise.dat');

%% Split the data into three partitions 
% Dtrn --> Training Set Data (60%)
% Dval --> Evaluation Set Data (20%)
% Dchk --> Test Set Data (20%)
[Dtrn,Dval,Dchk] = split_scale(D, 1);

% Display the sizes of each set
disp(['Training set size: ', num2str(size(Dtrn, 1))]);
disp(['Evaluation set size: ', num2str(size(Dval, 1))]);
disp(['Test set size: ', num2str(size(Dchk, 1))]);


%% Create ANFIS Models with Different Configurations

% Initialize an array to store options for generating FIS models
gen_options(1) = genfisOptions("GridPartition", ...
                            "InputMembershipFunctionType", "gbellmf", ...
                            "NumMembershipFunctions", 2, ...
                            "OutputMembershipFunctionType", "constant");
gen_options(2) = genfisOptions("GridPartition", ...
                            "InputMembershipFunctionType", "gbellmf", ...
                            "NumMembershipFunctions", 3, ...
                            "OutputMembershipFunctionType", "constant");
gen_options(3) = genfisOptions("GridPartition", ...
                            "InputMembershipFunctionType", "gbellmf", ...
                            "NumMembershipFunctions", 2);
gen_options(4) = genfisOptions("GridPartition", ...
                            "InputMembershipFunctionType", "gbellmf", ...
                            "NumMembershipFunctions", 3);

% Initialize a matrix to store performance metrics for each model
% Rows correspond to models, columns to [RMSE, NMSE, NDEI, R2]
performance_metrics = zeros(5,4); % Note: Only first 4 rows will be used


%% Train and Evaluate Each ANFIS Model

for i = 1:4
    % Generate an initial FIS model based on the current configuration
    model = genfis(Dtrn(:,1:end-1), Dtrn(:,end), gen_options(i));
    
    % Set ANFIS training options
    trn_options = anfisOptions("InitialFis", model, ...
                       "ValidationData", Dval, ...
                       "EpochNumber", 40);
    
    % Train the ANFIS model using the training data and options
    [trainFis, trainError, stepSize, bestValFis, valError] = anfis(Dtrn, trn_options);
    
    %% Plot Membership Functions Before and After Training
    
    figure(); % Create a new figure for membership functions
    
    for j = 1:5
        % Plot membership functions for the j-th input feature before training
        subplot(2,5,j);
        plotmf(model, 'input', j);
        title(['Before Training of Model ', num2str(i)]);
        xlabel(['Feature ', num2str(j)]);
        
        % Plot membership functions for the j-th input feature after training
        subplot(2,5,j+5);
        plotmf(bestValFis, 'input', j);
        title(['Feature ', num2str(j), ' of Best Model ', num2str(i)]);
        xlabel(['Feature ', num2str(j)]);
    end
    
    %% Plot Learning Curve
    
    figure(); % Create a new figure for learning curves
    
    % Plot training error
    subplot(2,1,1);
    plot(trainError, 'r');
    hold on;
    
    % Plot validation error
    plot(valError, 'g');
    hold off;
    
    % Customize the learning curve plot
    title(['Learning Curve of TSK Model ', num2str(i)]);
    xlabel('Iterations');
    ylabel('Error');
    legend('Training Error', 'Validation Error', 'Location', 'Best');
    
    %% Evaluate Model on Test Data
    
    % Predict the outputs for the test set using the trained FIS model
    prediction = evalfis(bestValFis, Dchk(:,1:end-1));
    
    % Calculate prediction error (actual - predicted)
    testError = Dchk(:,end) - prediction;
    
    % Plot the prediction error for each test sample
    subplot(2,1,2);
    plot(testError);
    title(sprintf("Prediction Error, Model %d", i));
    xlabel('Testing Data');
    ylabel('Prediction Error');

    
    %% Compute Performance Metrics
    
    % Calculate Root Mean Squared Error (RMSE)
    RMSE = sqrt(mse(prediction, Dchk(:,end)));
    
    % Calculate Coefficient of Determination (R²)
    R2 = 1 - sum((prediction - Dchk(:,end)).^2) / sum((Dchk(:,end) - mean(Dchk(:,end))).^2);
    
    % Calculate Normalized Mean Squared Error (NMSE)
    NMSE = 1 - R2;
    
    % Calculate Normalized Root Mean Squared Error (NDEI)
    NDEI = sqrt(NMSE);
    
    % Store the performance metrics for the current model
    performance_metrics(i, :) = [RMSE, NMSE, NDEI, R2];
end

% Display the performance metrics for all models
disp('Performance Metrics for Each Model:');
disp('Columns: [RMSE, NMSE, NDEI, R2]');
disp(performance_metrics);
