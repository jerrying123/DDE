% 
qrjsse_s = readmatrix('rdmd_error.csv');
dmdsse_s = readmatrix('edmd_error.csv');
%%
qrjsse = cumsum(qrjsse_s,2);
dmdsse = cumsum(dmdsse_s,2);

%%
T = linspace(1, 200, 200);
asse_qrj = mean(qrjsse,1);
asse_dmd = mean(dmdsse,1);


%%
% shading plots
max_j = max(qrjsse);
min_j = min(qrjsse);
max_aqr = max(dmdsse);
min_aqr = min(dmdsse);
%%
%koopman
% close all
hold on
shade(T,max_j,':g',T,min_j,':g', 'FillType',[1 2]);
shade(T,max_aqr,':b',T,min_aqr,':b', 'FillType',[1 2]);

jp = plot(T, asse_qrj, 'g', 'LineWidth', 3);
aqrp = plot(T, asse_dmd, '-.b', 'LineWidth', 3);

% legend([jp, bjp, aqrp, ap, rbfp], 'Joint with DE', 'Joint', 'Aggregate with DE', 'Aggregate', 'EDMD', 'Fontsize', 18)
legend([jp, aqrp], 'DDE', 'EDMD', 'Fontsize', 24)

ax = gca;
ax.FontSize = 24; 
% set(gca, 'YScale', 'log')
% axis([1,10,0,10000]);
xlabel('Time Steps', 'FontSize', 24);
ylabel('Sum Squared Error', 'FontSize', 24);
