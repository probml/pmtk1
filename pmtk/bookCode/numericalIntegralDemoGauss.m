%% Demo of how to compute functions of 1d/2d/3d Gaussian posterior using numerical integration

tol = 1e-3;
%% Now do it in 3d
mu3 = [5 1 -2]'; Sigma3 = 2*eye(3);
Zexact3 = sqrt(det(2*pi*Sigma3))

% picking the right range of integration is crucial...
x1min = -10; x1max = 20; x2min = -10; x2max = 20; x3min = -10; x3max = 10;

tic
target = @(X) gausspdfUnnormalized(X,mu3,Sigma3);
Zapprox3 = numericalIntegral(target, [x1min, x1max, x2min, x2max, x3min, x3max], tol)
toc


%% Now do it in 2d
mu2 = mu3(1:2); Sigma2 = Sigma3(1:2,1:2); 
Zexact2 = sqrt(det(2*pi*Sigma2))

tic
target = @(X) gausspdfUnnormalized(X,mu2,Sigma2);
Zapprox2 = numericalIntegral(target, [x1min, x1max, x2min, x2max])
toc

%% Now do it in 1d
mu1 = mu3(1); Sigma1 = Sigma3(1,1); 
Zexact1 = sqrt(det(2*pi*Sigma1))
tic
target = @(X) gausspdfUnnormalized(X,mu1,Sigma1);
Zapprox1 = numericalIntegral(target, [x1min, x1max])
toc


%% Posterior moments in 3d
target = @(X) gausspdfUnnormalized(X,mu3,Sigma3);
density = @(X) target(X)/Zapprox3;
mu = mu3; Sigma = Sigma3;
range = [x1min, x1max, x2min, x2max, x3min, x3max];

exact = 1; approx = 2;
tol = 1e-3;
tic
for dim=1:3
  postMean(exact,dim) = mu(dim);
  fn = @(X) X(:,dim);
  expfn = @(X) density(X) .* fn(X);
  postMean(approx,dim) =  numericalIntegral(expfn,range, tol);

  postMeanSquaredExact = Sigma(dim,dim) + mu(dim)^2;
  fn = @(X) X(:,dim).^2;
  expfn = @(X) density(X) .* fn(X);
  postMeanSquaredApprox =  numericalIntegral(expfn,range, tol);

  postVar(exact,dim) = Sigma(dim,dim);
  % E[X_1^2] = Var(X_1) + (E X_1)^2 
  postVar(approx,dim) = postMeanSquaredApprox - (postMean(approx,dim))^2;
end
toc



fprintf('Dim & Exact & Approx & Exact & Approx\n');
for d=1:3
fprintf('%d &  %5.3f & %5.3f & %5.3f & %5.3f\n', ...
  d, postMean(exact,d), postMean(approx,d), postVar(exact, d), postVar(approx, d));
end




fprintf('Mean & Exact & Approx & Quantity & Exact & Approx\n');
fprintf('E[X_1] & %5.3f & %.3f & E[X_2] & %5.3f & %5.3f\n', ...
  postMean(exact,1), postMean(approx,1), postMean(exact, 2), postMean(approx, 2));
fprintf('Var[X_1] & %5.3f & %.3f & Var[X_2] & %5.3f & %5.3f\n', ...
  postVar(exact,1), postVar(approx,1), postVar(exact, 2), postVar(approx, 2));


%% Posterior moments in 2d
target = @(X) gausspdfUnnormalized(X,mu2,Sigma2);
density = @(X) target(X)/Zapprox2;
mu = mu2; Sigma = Sigma2;

exact = 1; approx = 2;
tol = 1e-6;
tic
for dim=1:2
  postMean(exact,dim) = mu(dim);
  fn = @(X) X(:,dim);
  expfn = @(X) density(X) .* fn(X);
  postMean(approx,dim) =  numericalIntegral(expfn,[x1min,x1max,x2min,x2max], tol);

  % E[X_1^2] = Var(X_1) + (E X_1)^2 
  postMeanSquaredExact = Sigma(dim,dim) + mu(dim)^2;
  fn = @(X) X(:,dim).^2;
  expfn = @(X) density(X) .* fn(X);
  postMeanSquaredApprox =  numericalIntegral(expfn,[x1min,x1max,x2min,x2max], tol);

  postVar(exact,dim) = Sigma(dim,dim);
  postVar(approx,dim) = postMeanSquaredApprox - (postMean(approx,dim))^2;
end
toc
postMean
postVar







