
% Demo of fitting a Gamma distribution to the rainfall data used in Rice (1995) p383
X = dlmread('rainfallData.txt');
X = X'; X = X(:); % concatenate across rows, not columns
X = X(1:end-5); % removing trailing 0s

[a,b]= gamMOM(X);
[aMLE, bMLE] = gamMLE(X);

[v, binc] = hist(X);
h = binc(2)-binc(1);
N = length(X);
areaH = h*N;
figure(1);clf;bar(binc, v/areaH) 

%xs = linspace(min(X), max(X), 100);
%xs = linspace(binc(1), binc(end), 100);
xs = linspace(0.05,  binc(end), 100);

% Plot MoM
p = gampdf(xs, a, 1/b);
hold on
plot(xs, p, 'r-', 'linewidth', 3)
% Plot MLE
p = gampdf(xs, aMLE, 1/bMLE);
plot(xs, p, 'k:', 'linewidth', 3)
