% Define X and Y coordinates of the obstacle points
X_obstacle = [5, 5, 6, 6, 7, 7, 11];
Y_obstacle = [0, 1, 1, 2, 2, 3, 3];
figure;

% Plot the obstacle
plot(X_obstacle, Y_obstacle, 'black', 'LineWidth', 2);

% Set axis labels and title and axis limits
xlabel('X');   
ylabel('Y');
title('Car Route');
xlim([0, 11]);
ylim([0, 4]);

hold on;

% Initial Point Coordinates
Xinit = 4.1;
Yinit = 0.3;

% Plot the Initial Position point
plot(Xinit, Yinit, 'ro', 'MarkerFaceColor', 'g', 'MarkerSize', 5);

% Desired point coordinates
Xd = 10;
Yd = 3.2;

% Plot the Desired Position point
plot(Xd, Yd, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 5);

% Plot the (X-Y) car route
plot(out.X.Data, out.Y.Data, '--blue', 'LineWidth', 1.5);

% Plot and Display the Final Point of the Route
final_X = out.X.Data(length(out.X.Data));
final_Y = out.Y.Data(length(out.Y.Data));

plot(final_X, final_Y, 'ro', 'MarkerFaceColor', 'y', 'MarkerSize', 5);

fprintf("Final Point: (%d, %d)", round(final_X,4), round(final_Y,4));

% Add a legend for the obstacle, initial point, desired point, car route, and final point
legend('Obstacle', 'Initial Position', 'Desired Position', 'Car Route', 'Final Position', ...
    'Location', 'northwest');


