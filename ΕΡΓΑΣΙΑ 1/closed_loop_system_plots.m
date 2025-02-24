% Plot reference input and output
figure();
plot(out.r.Time, out.r.Data);
hold on;
plot(out.y.Time, out.y.Data);
title('System Response: ke = 5.5, ki = 1.5, k = 10');
xlabel('Time');
ylabel('Velocity');

% Calculate the rise time of the signal
rise = risetime(out.y.Data, out.y.Time);
fprintf("\n\nRise Time(sec): %d", rise);

% Calculate the overshoot factor
os = overshoot(out.y.Data, out.y.Time);
os = (max(out.y.Data)-out.y.Data(end,:))/out.y.Data(end,:)*100;
fprintf("\nOvershoot factor(%%): %d", os);

