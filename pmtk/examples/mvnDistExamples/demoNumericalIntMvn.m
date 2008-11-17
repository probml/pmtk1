%% Demo of computing functions of MVN posterior using numerical integration

function demoNumericalInt()

setSeed(1);
d = 2;

muTrue = [5 1]'; STrue = 2*eye(2);
X = sample(MvnDist(muTrue, STrue), 5);
Strue = STrue;

m0 = zeros(d,1); S0 = 10*eye(d);
prior = MvnDist(m0, S0);
model = fit(MvnDist(prior, STrue), 'data', X);
post = model.mu;
logZ = lognormconst(post)
%logZtrue = 0.5*logdet(post.Sigma) + (d/2)*log(2*pi)
Z = exp(logZ)

x1min = -10; x1max = 10; x2min = x1min; x2max = 10;

tic
Zapprox = dblquad(@postUnnormalized, x1min, x1max, x2min, x2max)
toc

%{
foo = @(x1,x2) exp(logprob(post, [x1 x2]) + logZ);
tic
Zapprox2 = dblquadvec(foo,x1min,x1max,x2min,x2max)
toc
%}

dim = 1;
% E[X_1^2] = Var(X_1) + (E X_1)^2 = 27
postMeanSquaredExact = Strue(dim,dim) + muTrue(dim)^2

tic
foo = @(x1,x2) postMean(x1,x2, dim, 2, Z);
postMeanSquaredApprox =  dblquad(foo,x1min,x1max,x2min,x2max)
toc

postMeanExact = muTrue(dim)
foo = @(x1,x2) postMean(x1,x2, 1, 1, Z);
postMeanApprox =  dblquad(foo,x1min,x1max,x2min,x2max)

postVarExact = Strue(1,1)
postVarApprox = postMeanSquaredApprox - (postMeanApprox)^2

keyboard

%{
tic
foo = @(x1,x2) exp(logprob(post, [x1 x2]))*(x1^2);
moment1approx2 =  dblquadvec(foo,x1min,x1max,x2min,x2max)
toc
%}

% nested functions

  function prob = postUnnormalized(x1,x2)
    % prob(i) = p(x1(i),x2, data)
    n = length(x1);
    x2 = x2*ones(n,1);
    prob =  exp(logprob(post, [x1(:) x2(:)]) + logZ);
  end

  function f = postMean(x1,x2, dim, pow, Z)
    % f(i) = xdim(i)^pow * p(x1(i),x2 | data)
    prob =  postUnnormalized(x1, x2)/Z;
    if dim==1
     x = x1(:);
    else
      n = length(x1);
      x = x2*ones(n,1);
    end
     f = x.^pow .* prob;
  end

  function f = postMean2(x1,x2, dim, pow)
    % f(i) = xdim(i)^pow * p(x1(i),x2 | data)
    n = length(x1);
    x2 = x2*ones(n,1);
    prob =  exp(logprob(post, [x1(:) x2(:)]));
    if dim==1
     x = x1(:);
    else
      x = x2(:);
    end
     f = x.^pow .* prob;
  end

end

