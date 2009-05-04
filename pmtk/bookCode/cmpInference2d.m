%% Demo of approximate inference on 2d distributions

%{
% Johnson and Albert p35
logtarget = @(X) smokingCancerLogpost(X);
target = @(X) exp(logtarget(X));
range1 = -1:0.1:6; range2 = 2:0.1:10; 
%}


mu = [5 1]'; 
Sigma = [2 1; 1 1.5]; 
range1 = 0:0.1:10; range2 = -5:0.05:5;
truedist = MvnDist(mu,Sigma);

%{
prior = NormInvGammaDist(0,0.01,0.01,0.01);
model = GaussDist(prior);
X = [-1 0 2 3.5]';
model = fit(model, X);
truedist = model.mu; % posterior over params
range1 = -4:0.1:6; range2 = 0.1:0.1:6;
%}


figure; plot(truedist); title('exact');

initVal = [mean(range1) mean(range2)];
logtarget = @(X) logprob(truedist, X, false); 
logZexact = lognormconst(truedist);
muExact = mean(truedist);
varExact = var(truedist);
modeExact = mode(truedist);
Sexact = sample(truedist, 1000);


% Sample approximation
psamp = SampleBasedDist(Sexact);
plot(psamp);title('exact samples')
set(gca,'xlim',[min(range1) max(range1)]);
set(gca,'ylim',[min(range2) max(range2)]);

muSample = mean(psamp);
varSample = var(psamp);
modeSample = NaN;
logZsample = NaN;

% Grid method
tic
pgrid  = GridDist(logtarget, range1, range2);
logZgrid = lognormconst(pgrid);
muGrid = mean(pgrid);
varGrid = var(pgrid);
modeGrid = mode(pgrid);
toc
figure; plot(pgrid, 'type', 'contour'); title('grid approximation')
set(gca,'xlim',[min(range1) max(range1)]);
set(gca,'ylim',[min(range2) max(range2)]);

% Laplace approx
tic
%plaplace = LaplaceApproxDist(logtarget, initVal);
[mu,Sigma,logZlaplace] = laplaceApproxNumerical(logtarget, initVal);
plaplace = MvnDist(mu,Sigma);
logZlaplace =  lognormconst(plaplace);
muLaplace = mean(plaplace);
varLaplace = var(plaplace);
modeLaplace = mode(plaplace);
toc
figure; plot(plaplace); title('Laplace approximation')
set(gca,'xlim',[min(range1) max(range1)]);
set(gca,'ylim',[min(range2) max(range2)]);


% Numerical integration
tic
pnum = NumIntDist(logtarget, [min(range1) max(range1) min(range2) max(range2)]);
logZnum =  lognormconst(pnum);
muNum = mean(pnum);
varNum = var(pnum);
modeNum = mode(pnum);
toc

fprintf('Mean & Grid & Laplace & Numerical & Exact & Sample\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f & %5.3f & %5.3f & %5.3f\n',...
  d, muGrid(d), muLaplace(d), muNum(d), muExact(d), muSample(d));
end
fprintf('Var & Grid & Laplace & Numerical & Exact & Sample\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f & %5.3f & %5.3f & %5.3f\n', ...
  d, varGrid(d), varLaplace(d), varNum(d), varExact(d), varSample(d));
end
fprintf('Mode & Grid & Laplace & Numerical & Exact\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f & %5.3f & %5.3f\n', ...
  d, modeGrid(d), modeLaplace(d), modeNum(d), modeExact(d));
end
fprintf('logZ & Grid & Laplace & Numerical & Exact\n');
fprintf(' %5.3f & %5.3f & %5.3f %5.3f\n', ...
  logZgrid, logZlaplace, logZnum, logZexact);

%{
fprintf('Mean & Grid & Laplace & Numerical\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f & %5.3f\n', d, muGrid(d), muLaplace(d), muNum(d));
end
fprintf('Var & Grid & Laplace & Numerical\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f & %5.3f\n', d, varGrid(d), varLaplace(d), varNum(d));
end
fprintf('Mode & Grid & Laplace & Numerical\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f & %5.3f\n', d, modeGrid(d), modeLaplace(d), modeNum(d));
end
fprintf('logZ & Grid & Laplace & Numerical\n');
fprintf(' %5.3f & %5.3f & %5.3f\n', logZgrid, logZlaplace, logZnum);
%}


%{
% Credible interval for alpha
z = norminv(1-0.025);
sigma = sqrt(C(1,1));
alphaHat = mu(1);
credibleInterval = [alphaHat - z*sigma, alphaHat + z*sigma]
ppos = normcdf(alphaHat/sigma) % Prob(alpha>0)
%}



