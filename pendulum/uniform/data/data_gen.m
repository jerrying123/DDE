% Scan of initial conditions that determines the initial conditions that
% have stable and unstable trajectories.
xlim1 = 0.8;
xlim2 = 2;
N_data = 500;
X = [];
mu = [0,0];

dt = 0.1;
tspan = [0 dt/2 dt];
nxs = [10, 15, 25, 30,35,40,45, 50,60, 100, 150];
datasizes = nxs.^2;
for k = datasizes
    nx = sqrt(k);
    Xt_s = [];
    Xtp1_s = [];
    x1 = linspace(-xlim1,xlim1,nx);
    x2 = linspace(-xlim2,xlim2,nx);
    for i = 1:nx
        for j = 1:nx
            x0 = [x1(i), x2(j)];
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
    
    Xt_agg = Xt_s;
    Xtp1_agg = Xtp1_s;
    
    %%
    % undivided sets
    writematrix(Xt_agg,'agg_t_'+string(length(Xt_agg))+'.csv');
    writematrix(Xtp1_agg,'agg_t1_'+string(length(Xt_agg))+'.csv');
end
