% 
qrjsse = readmatrix('rdmd_error.csv');
dmdsse = readmatrix('edmd_error.csv');


%%
T = linspace(1, 200, 200);
asse_qrj = mean(qrjsse,1);
asse_dmd = mean(dmdsse,1);

%%
asse_qrj(100)
asse_dmd(100)