load ICs.mat
tspan = [0 5];

figure;
for i = 1:length(X)
    x0 = X(i,:);
%     tspan = [0 10];
    [t,x_o] = ode45(@(t,x)pend(t,x),tspan, x0);
    if boolArr(i) == true
        plot(x_o(:,1),x_o(:,2),'g');
    else
        plot(x_o(:,1),x_o(:,2),'r');
    end
    hold on
end
