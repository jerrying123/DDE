%Same as f_nonlinear above, but has inclusion of t input for the use of
%ode45
function dxdt = sys_model(t,x)
dxdt = f_nonlinear(x);
end
function dxdt = f_nonlinear(x)
gain = 19;
dgain = 5;
x0 = x;
xw1 = [0,0];
xw2 = [2,0];
k = 1699;
r = 0.3;
I = 0.45;
g = -9.8;
m = 10;
xf = [0.8 -4 0 0 4.1145 0];
u = -gain*(xf(5) - x0(5));
d1 = calc_d(x, xw1);
d2 = calc_d(x, xw2);
h = find_eta(x, m, xw1, xw2, k);
dxdt = zeros(size(x));
dxdt(1) = x(3); % x velocity of pam
dxdt(2) = x(4); % y velocity of pam
dxdt(3) = (h(1))/m; % x acceleration
dxdt(4) = (h(2)+ m*g)/m; % y acceleration
dxdt(5) = x(6); % winch cable a velocity

if norm(d1) < x(5)
    dxdt(6) = -r/I*(u(1));
else
    dxdt(6) = -r/I*(u(1) - r * sqrt(h(1)^2+h(2)^2)); 
end
end
function d = calc_d(x, xw)

% takes in winch position and mass state and calculates the distance
% between the two
dx = xw(1) - x(1);
dy = xw(2) - x(2);
d = [dx, dy];
end