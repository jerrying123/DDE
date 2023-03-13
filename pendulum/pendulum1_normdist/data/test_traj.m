close all
x0 = [0    -5];
tspan = [0 10];
[t,x_o] = ode45(@(t,x)pend(t,x),tspan, x0);

plot(t, x_o(:,1), 'b')
hold on;
plot(t, x_o(:,2), 'r')