x0 = 57.5493931057184
Xt = [];
Xtp1 = [];
tspan = [0 0.05 0.1];


for i = 1:100
       
    Xt = [Xt; x0];
    [t,x] = ode45(@(t,x)oneord(t,x), tspan, x0);
    t
    x0 = x(end,:);
    Xtp1 = [Xtp1; x0];        
            
end