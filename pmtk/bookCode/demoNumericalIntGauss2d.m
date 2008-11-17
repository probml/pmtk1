
%% Demo of how to compute functions of 2d Gaussian posterior using numerical integration

d = 2;
seed = 0; randn('state', seed);
mu = [5 1]'; Sigma = 2*eye(2);
logZexact = 0.5*logdet(Sigma) + (d/2)*log(2*pi);
Zexact = exp(logZexact)

% picking the right range of integration is crucial...
x1min = -10; x1max = 20; x2min = -10; x2max = 20;

tic
target = @(X) gausspdfUnnormalized(X,mu,Sigma);
Zapprox = dblquadrep(target, [x1min, x1max, x2min, x2max])
toc

density = @(X) target(X)/Zapprox;

dim = 1;

postMeanExact = mu(dim)
fn = @(X) X(:,dim);
expfn = @(X) density(X) .* fn(X);
postMeanApprox =  dblquadrep(expfn,[x1min,x1max,x2min,x2max])

% E[X_1^2] = Var(X_1) + (E X_1)^2 = 27
postMeanSquaredExact = Sigma(dim,dim) + mu(dim)^2
fn = @(X) X(:,dim).^2;
expfn = @(X) density(X) .* fn(X);
postMeanSquaredApprox =  dblquadrep(expfn,[x1min,x1max,x2min,x2max])

postVarExact = Sigma(dim,dim)
postVarApprox = postMeanSquaredApprox - (postMeanApprox)^2




