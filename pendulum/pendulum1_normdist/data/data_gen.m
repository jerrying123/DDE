load ICs.mat
dt = 0.1;
tspan = [0 dt/2 dt];
T_tr = dt;
Xt_s = [];
Xtp1_s = [];
Xt_u = [];
Xtp1_u = [];



for i = 1:length(X)
    x0 = X(i,:);
    for j = 1:floor(T_tr/tspan(end))
        Xt_s = [Xt_s; x0];
        [t_arr,x] = ode45(@(t,x)pend(t,x), tspan, x0);
        x0 = x(end,:);
        Xtp1_s = [Xtp1_s; x0];        
        if t_arr(end) < tspan(end)
            Xt_s(end,:) = [];
            Xtp1_s(end,:) = [];
        end
    end
end

Xt_agg = [Xt_s];
Xtp1_agg = [Xtp1_s];

%%
% undivided sets
writematrix(Xt_s,'stable_t.csv');
writematrix(Xtp1_s,'stable_t1.csv');

writematrix(Xt_u,'unstable_t.csv');
writematrix(Xtp1_u,'unstable_t1.csv');

writematrix(Xt_agg,'agg_t.csv');
writematrix(Xtp1_agg,'agg_t1.csv');
