% Scan of initial conditions that determines the initial conditions that
% have stable and unstable trajectories.
xlim1 = 0.2;
xlim2 = 2;
x1 = linspace(-xlim1,xlim1,nx1);
x2 = linspace(-xlim2,xlim2,nx2);
N_data = 500;
X = [];
mu = [0,0];
sigma = [xlim1/2 0; 0 xlim2/2];
R = chol(sigma);
for i=1:N_data
    x0 = [mu' + R*randn(2,1)]';
    X = [X; x0];
end
save('ICs.mat', 'X');
%%
x1 = linspace(-xlim1,xlim1,nx1);
x2 = linspace(-xlim2,xlim2,nx2);
N_data = 500;
X = [];

for i=1:N_data
    x0 = [mu' + R*randn(2,1)]';
    X = [X; x0];
       
end
save('ICs2.mat', 'X');