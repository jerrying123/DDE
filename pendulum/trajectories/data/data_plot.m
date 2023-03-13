T = readtable('agg_t_10000.csv');
plot(T.Var1, T.Var2,'o')
xlabel('Angle (rad)', 'fontsize',20);
ylabel('Angular Velocity (rad/s)', 'fontsize', 20);