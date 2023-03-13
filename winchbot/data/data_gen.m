load ICs.mat

tspan = [0 0.05 0.1];
T_tr = 10;
Xt = [];
Xtp1 = [];

for i = 1:length(X)
    x0 = X(i,:);
    for j = 1:floor(T_tr/tspan(end))
        Xt = [Xt; x0];
        [t_arr,x] = ode45(@(t,x)sys_model(t,x), tspan, x0);
        x0 = x(end,:);
        Xtp1 = [Xtp1; x0];        
        if t_arr(end) < tspan(end)
            Xt(end,:) = [];
            Xtp1(end,:) = [];
        end
    end
end



%%
% undivided sets


writematrix(Xt,'agg_t.csv');
writematrix(Xtp1,'agg_t1.csv');
