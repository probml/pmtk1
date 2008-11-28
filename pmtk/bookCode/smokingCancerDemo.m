%% Demo of approximate inference on 2d cancer smoking example

% Johnson and Albert p35
logtarget = @(X) smokingCancerLogpost(X);
range1 = -1:0.1:6; range2 = 2:0.1:10; 
initVal = [mean(range1) mean(range2)];

% Exact samples from the exact posterior
pLpost = BetaDist(83,3);
pCpost = BetaDist(72,14);
N = 1000;
S = zeros(N,2);
L = 1; C = 2;
S(:,L) = sample(pLpost,N);
S(:,C) = sample(pCpost,N);
alpha = log( (S(:,L)./(1-S(:,L)))./(S(:,C)./(1-S(:,C))));
eta = log( (S(:,L)./(1-S(:,L))) .* (S(:,C)./(1-S(:,C))));
Sexact = [alpha(:) eta(:)];

% Sample approximation
psamp = SampleDist(Sexact);
plot(psamp);title('exact samples')
set(gca,'xlim',[min(range1) max(range1)]);
set(gca,'ylim',[min(range2) max(range2)]);
grid on

muSample = mean(psamp);
varSample = var(psamp);
modeSample = NaN;
logZsample = NaN;
m = marginal(psamp,1);
[credSample(1), credSample(2)] = credibleInterval(m);
pposSample = 1-cdf(m,0);

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
grid on

% Laplace approx
tic
plaplace = LaplaceApproxDist(logtarget, initVal);
logZlaplace =  lognormconst(plaplace);
muLaplace = mean(plaplace);
varLaplace = var(plaplace);
modeLaplace = mode(plaplace);
m = convertToScalarDist(marginal(plaplace,1));
[credLaplace(1), credLaplace(2)] = credibleInterval(m);
pposLaplace = 1-cdf(m,0);
toc
figure; plot(plaplace); title('Laplace approximation')
set(gca,'xlim',[min(range1) max(range1)]);
set(gca,'ylim',[min(range2) max(range2)]);
grid on

% Numerical integration
tic
pnum = NumIntDist(logtarget, [min(range1) max(range1) min(range2) max(range2)]);
logZnum =  lognormconst(pnum);
muNum = mean(pnum);
varNum = var(pnum);
modeNum = mode(pnum);
toc


fprintf('Mean & Grid & Laplace & Numerical & Sample\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f & %5.3f & %5.3f\n', ...
  d, muGrid(d), muLaplace(d), muNum(d), muSample(d));
end
fprintf('Var & Grid & Laplace & Numerical & Sample\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f & %5.3f & %5.3f\n', ...
  d, varGrid(d), varLaplace(d), varNum(d), varSample(d));
end
fprintf('Mode & Grid & Laplace & Numerical\n');
for d=1:2
fprintf('%d &  %5.3f & %5.3f & %5.3f\n', ...
  d, modeGrid(d), modeLaplace(d), modeNum(d));
end
fprintf('logZ & Grid & Laplace & Numerical\n');
fprintf(' %5.3f & %5.3f & %5.3f\n', logZgrid, logZlaplace, logZnum);


fprintf('\n');
% Reproduce table on p52 of Johnson & Albert
fprintf('Method & E[alpha] & SD(alpha) & 95pc interval & ppos\n');
fprintf('Laplace & %5.3f & %5.3f & (%3.2f,%3.2f) & %5.3f\n', ...
  muLaplace(1), sqrt(varLaplace(1)), credLaplace(1), credLaplace(2), pposLaplace);
fprintf('Grid & %5.3f & %5.3f \n', ...
  muGrid(1), sqrt(varGrid(1)));
fprintf('NumInt & %5.3f & %5.3f \n', ...
  muNum(1), sqrt(varNum(1)));
fprintf('Sample & %5.3f & %5.3f & (%3.2f,%3.2f) & %5.3f\n', ...
  muSample(1), sqrt(varSample(1)), credSample(1), credSample(2), pposSample);




