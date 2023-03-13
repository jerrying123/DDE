x1 = linspace(-2,2,25);
x2 = linspace(-2,2,25);
tspan = [0 0.05 0.1];
Xt = [];
Xtp1 = [];


for i = 1:length(x1)
    for j = 1:length(x2)
        x0 = [x1(i), x2(j)];
           
        Xt = [Xt; x0];
        [t_arr,x] = ode45(@(t,x)twoord(t,x), tspan, x0);
        x0 = x(end,:);
        Xtp1 = [Xtp1; x0];        
        if t_arr(end) < 0.1
            Xt(end,:) = [];
            Xtp1(end,:) = [];
        end
    end
            
end


%%
% undivided sets
writematrix(Xt,'qr_t.csv');
writematrix(Xtp1,'qr_t1.csv');


