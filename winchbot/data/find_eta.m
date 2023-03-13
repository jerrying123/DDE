function h = find_eta(x, m, xw1, xw2, k)
%Calculates eta based on state variables.
%x(1) is x, x(2) is xdot
g = -9.8;
h = zeros(4, 1);
d1 = calc_d(x, xw1);
L1 = x(5);
if norm(d1) < L1
    h(1) = 0;
    h(2) = 0;
else
    f1 = k*(norm(d1) - L1); %force
    h(1) = f1*d1(1)/norm(d1);
    h(2) = f1*d1(2)/norm(d1);
end

end
function d = calc_d(x, xw)

dx = xw(1) - x(1);
dy = xw(2) - x(2);
d = [dx, dy];
end