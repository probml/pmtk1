%%  Posterior Predictive Distribution for Beta-Binomial model
%#testPMTK
N = 10;
X = [1 2]';
N1 = sum(X);
N0 = N*length(X)-N1;
a = 2;
b = 2;
%% Create the model
m = BinomConjugate('-N',N,'-prior', BetaDist(a,b));
%% Prior Predictive
m0 = marginalizeOutParams(m);                   % BetaBinomDist object
figure; plot(m0); title('prior predictive')
if doPrintPmtk, printPmtkFigures('BBpriorpred'); end;
%% Posterior Predictive
m = fit(m,  X);
mm = marginalizeOutParams(m);                   % posterior predictive is BetaBinomDist
figure;
h1 = plot(mm, 'plotArgs', 'b'); title('posterior predictive')
%% MAP estimation
m3 = fit(BinomDist('-N',N, '-prior',BetaDist(a,b)), X);
hold on
h3 = plot(m3);
set(h3, 'faceColor','r','barwidth',0.5);
legend('postpred', 'plugin');
if doPrintPmtk, printPmtkFigures('BBpostpred'); end;
