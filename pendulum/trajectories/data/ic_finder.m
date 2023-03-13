% Scan of initial conditions that determines the initial conditions that
% have stable and unstable trajectories.
xlim1 = 0.2;
xlim2 = 2;
nx1 = 10;
nx2 = 10;
x1 = linspace(-xlim1,xlim1,nx1);
x2 = linspace(-xlim2,xlim2,nx2);
tspan = [0 10];
X = [];
boolArr = [];

for i=1:length(x1)
   for j = 1:length(x2)
       x0 = [x1(i), x2(j)];
       [t,x_o] = ode45(@(t,x)pend(t,x),tspan, x0);
       X = [X; x0];
       xend = x_o(end,:);

       if abs(xend(1)) <= abs(x0(1)) || abs(xend(2)) <= abs(x0(2)) 
           staBool = true;
       else
           staBool = false;
       end
       
       boolArr = [boolArr; staBool];
   end
end

save('ICs.mat', 'boolArr', 'X');

%%
xlim1 = 0.7;
xlim2 = 2;
x1 = linspace(-xlim1,xlim1,nx1);
x2 = linspace(-xlim2,xlim2,nx2);
tspan = [0 10];
X = [];
boolArr = [];

for i=1:length(x1)
   for j = 1:length(x2)
       x0 = [x1(i), x2(j)];
       [t,x_o] = ode45(@(t,x)pend(t,x),tspan, x0);
       X = [X; x0];
       xend = x_o(end,:);

       if abs(xend(1)) <= abs(x0(1)) || abs(xend(2)) <= abs(x0(2)) 
           staBool = true;
       else
           staBool = false;
       end
       
       boolArr = [boolArr; staBool];
   end
end

save('ICs2.mat', 'boolArr', 'X');