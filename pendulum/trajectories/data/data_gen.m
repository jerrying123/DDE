load ICs.mat
xlim1 = 0.7;
xlim2 = 2;
x1 = linspace(-xlim1, xlim1, 10);
x2 = linspace(-xlim2, xlim2, 10);
dt = 0.1;
tspan = [0 dt/2 dt];
datasizes = [250,350,500,700,800,1000,1500,2500,5000,10000,25000];
for k = datasizes
    T_tr = (k)*dt/length(X);%(k-100)*dt/length(X);
    Xt_s = [];
    Xtp1_s = [];
    for i = 1:length(X)
        x0 = X(i,:);
        for j = 1:T_tr/tspan(end)
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
%     for i = 1:10
%         for j = 1:10
%             x0 = [x1(i), x2(j)];
%             Xt_s = [Xt_s; x0];
%             [t_arr,x] = ode45(@(t,x)pend(t,x), tspan, x0);
%             x0 = x(end,:);
%             Xtp1_s = [Xtp1_s; x0];  
%         end
%     end
    length(Xt_s)
    Xt_agg = Xt_s;
    Xtp1_agg = Xtp1_s;
    
    %%
    % undivided sets
    writematrix(Xt_agg,'agg_t_'+string(length(Xt_agg))+'.csv');
    writematrix(Xtp1_agg,'agg_t1_'+string(length(Xt_agg))+'.csv');
end
