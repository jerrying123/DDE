% Scan of initial conditions that determines the initial conditions that
% have stable and unstable trajectories.
Num_trials = 100;
X = [];
cableAmp = 1.5;
cableVar = 0.25;
zVar = 0.25;
for p = 1:Num_trials

    if floor(p/Num_trials) == 0
        % nothing is in tension
        L = rand()*cableVar+cableAmp;
        x_ = 0.25*rand()+1;
        z = 0 - sqrt(L^2 - x_^2)+zVar; 
        pos = [x_ z]';
        cable_L = L+zVar;
    elseif floor(p/Num_trials) == 1
        % cable a is in tension
        L = rand()*cableVar+cableAmp;
        x_ = 0.25*rand()+1;
        z = 0 - sqrt(L^2 - x_^2); 
        pos = [x_ z]';
        cable_L = L;
    end
    vel = [0 0];
    x0 = [pos' vel cable_L' 0]';
    X = [X; x0'];
end


save('ICs2.mat', 'X');
writematrix(X, 'ICs2.csv');