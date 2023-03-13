x1s = linspace(-2.5,1,5);
x2s = linspace(-2.5,1,5);
tspan = [0 0.2];

figure;
for i = 1:length(x1s)
    x0 = [x1s(i)    x2s(i)];
    tspan = [0 10];
    [t,x_o] = ode45(@(t,x)twoord(t,x),tspan, x0);
    if i <5
        plot3(x_o(:,1),x_o(:,2),t);
    end
    hold on
end
legend()