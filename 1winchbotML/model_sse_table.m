% qrjsse = readmatrix('qrj_error.csv');
% aqrsse = readmatrix('a_qr_error.csv');
% asse = readmatrix('a_error.csv');
% bsse = readmatrix('bj_error.csv');
% rbfsse = readmatrix('rbf_error.csv');

% 
qrjsse = readmatrix('qrj_error_u.csv');
aqrsse = readmatrix('a_qr_error_u.csv');
asse = readmatrix('a_error_u.csv');
bsse = readmatrix('bj_error_u.csv');
rbfsse = readmatrix('rbf_error_u.csv');

% qrjsse = readmatrix('qrj_error_s.csv');
% aqrsse = readmatrix('a_qr_error_s.csv');
% asse = readmatrix('a_error_s.csv');
% bsse = readmatrix('bj_error_s.csv');
% rbfsse = readmatrix('rbf_error_s.csv');


%%
T = linspace(1, 20, 20);
asse_qrj = mean(qrjsse,1);
asse_aqr = mean(aqrsse,1);
asse_a = mean(asse,1);
asse_bj = mean(bsse,1);
asse_rbf = mean(rbfsse,1);

%%
asse_bj(5)
asse_qrj(5)
asse_a(5)
asse_aqr(5)
asse_rbf(5)