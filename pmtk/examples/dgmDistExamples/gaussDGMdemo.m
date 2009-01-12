%%  Koller & Friedman Gaussian DGM Example
% p233
G = zeros(3,3);
G(1,2) = 1; G(2,3)=1;
% LinGaussCPD(w, w0, sigma2)
CPDs{1} = LinGaussCPD([], 1, 4);
CPDs{2} = LinGaussCPD(0.5, -5, 4);
CPDs{3} = LinGaussCPD(-1, 4, 3);
dgm = DgmDist(G, 'CPDs', CPDs, 'infEng', GaussInfEng());
%p = predict(dgm, 2, -3.1, [1 3]);
tmp = condition(dgm, 2, -3.1);
query = marginal(tmp, [1 3])
X = sample(dgm, 1000);
dgm2 = DgmDist(G);
dgm2 = mkRndParams(dgm2, 'CPDtype', 'LinGaussCPD');
dgm2 = fit(dgm2, X);
for j=1:3
    fprintf('node %d, orig: w %5.3f, w0 %5.3f, v %5.3f, est w %5.3f, w0 %5.3f, v %5.3f\n',...
        j, dgm.CPDs{j}.w, dgm.CPDs{j}.w0, dgm.CPDs{j}.v,...
        dgm2.CPDs{j}.w, dgm2.CPDs{j}.w0, dgm2.CPDs{j}.v);
end
Xtest = sample(dgm, 100);
[n d] = size(Xtest);
M = rand(n,d)>0.8;
L = sum(logprob(dgm, Xtest, 'interventionMask', M))
