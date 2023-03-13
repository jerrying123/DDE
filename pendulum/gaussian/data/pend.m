function dxdt = pend(t,x)
k = 200; %stiffness of the wall
c = 1; %linear damping coefficient
dx1 = x(2);
Fk = 0;
if abs(x(1)) > pi/4
    Fk = -sign(x(1))*k*(abs(x(1)) - pi/4)^2;
end
dx2 = -sin(x(1)) + Fk - sign(x(2))*c*x(2)^2; %g/l is set to 1

dxdt = [dx1; dx2];
end
