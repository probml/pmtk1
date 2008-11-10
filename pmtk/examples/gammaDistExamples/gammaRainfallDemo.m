%% Rainfall Demo
% Fitting a Gamma distribution to the rainfall data used in Rice (1995) p383
X = dlmread('rainfallData.txt');
X = X'; X = X(:); % concatenate across rows, not columns
X = X(1:end-5); % removing trailing 0s
objMoM = inferParams(GammaDist, 'data', X, 'method', 'mom');
objMLE = inferParams(GammaDist, 'data', X, 'method', 'mle');
[v, binc] = hist(X);
h = binc(2)-binc(1);
N = length(X);
areaH = h*N;
figure(1);clf;bar(binc, v/areaH);hold on
xs = [0.05,  binc(end)];
h(1)=plot(objMoM, 'xrange', xs, 'plotArgs', {'r-', 'linewidth', 3});
h(2)=plot(objMLE, 'xrange', xs, 'plotArgs', {'k:', 'linewidth', 3});
legend(h, 'MoM', 'MLE')