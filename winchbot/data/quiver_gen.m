close all
x = linspace(-2,2,15);
y = linspace(-2,2,15);
[X,Y] = meshgrid(x,y);

U = zeros(size(X));
V = zeros(size(Y));
for i = 1:length(x)
    for j = 1:length(y)
        dxdt = twoord([X(i,j),Y(i,j)]);
        U(i,j) = dxdt(1)*5;
        V(i,j) = dxdt(2)*5;
    end
end
%stable fixed
x1 = 1 + sqrt(19)/3;
y1 = -11/9 - 2/9*sqrt(19);
x2 = 1 - sqrt(19)/3;
y2 = -11/9 + 2/9*sqrt(19);
%unstable
x3 = 2/3;
y3 = 1;
%saddle
x4 = 0;
y4 = 0;
figure
quiver(X,Y,U,V)
hold on
% stabp = plot([x1, x2], [y1, y2], '*g');
% unstp = plot(x3,y3,'xr');
% sadp = plot(x4,y4,'ob');
xlabel('x','FontSize',15)
ylabel('y', 'Fontsize',15)
title('Quiver Plot of 2nd Order ODE', 'FontSize',15)
% legend([stabp, unstp, sadp], 'Stable Fixed Point','Unstable Fixed Point', 'Saddle Point', 'Fontsize', 15)
%%
dxdt = twoord([2.45296631451, -2.1908664319]);
dxdt = twoord([-0.45296631451, -0.25357801254]);

function dxdt = twoord(x)
y = x(2);
x = x(1);
xdot = -x + x^2 + y^2;
ydot = -y + y^2 + x^2 - x;
dxdt = [xdot; ydot];
end
