classdef LogregBinaryGibbsAuxFitEng < OptimEng
 % Auxiliary variable Gibbs sampling for 
% Binary logistic regression with Gaussian prior
% We use the algorithm of
% Holmes and Held, "Bayesian auxiliary variable models for binary
% and multinomial regression", Bayesian Analysis, 1(1):145-168, 2006

%#author Mark Schmidt

properties
  nsamples;
  verbose;
  scheme; % 1=w,z then lambda, 2 = z,lambda then w
end


%% Main methods
methods
  
  function m = LogregBinaryGibbsAuxFitEng(varargin)
    % LogregBinaryMc(nsamples, verbose, scheme)
    [m.nsamples, m.verbose, m.scheme] = ...
      processArgs( varargin ,...
      '-nsamples', 1000, ...
      '-verbose', false, ...
      '-scheme', 1);
  end
  
  function [model, out] = fit(eng, model, data)
    % m = fit(eng, m, D) Compute posterior estimate
    % D is DataTable containing:
    % X(i,:) is i'th input; do *not* include a column of 1s
    % y(i) is i'th response
   X = data.X; y = data.y;
   y = canonizeLabels(y); % 1,2
   y = y-1; % 0,1
   assert(isa(model.prior, 'MvnDist'))
   v = model.prior.Sigma;
   switch eng.scheme
     case 1,
       samples = logist2Sample1(X,y,v,eng.nsamples, eng.verbose);
     case 2,
       samples = logist2Sample2(X,y,v,eng.nsamples, eng.verbose);
     otherwise
       error('unknown scheme')
   end
    model.wDist.wsamples = samples;
   out = [];
  end
  
  
end % methods


end


%% scheme 1
function [beta] = logist2Sample1(X,y,v,numSamples,verbose)

[n,p] = size(X);

% Initialize Mixing Weights (we use a vector rather than matrix in paper)
Lam = ones(n,1);

% Draw initial Z from truncated Normal
Z = abs(randn(n,1)).*sign(y-.5);

for i = 1:numSamples
    if verbose, fprintf('Drawing sample %d\n',i);end

    % v and V should be psd, so no need to use slow pinv
    V = (X'*diag(Lam.^-1)*X + v^-1)^-1;
    L = chol(V)';

    S = V*X';
    B = S*diag(Lam.^-1)*Z;

    for j = 1:n
        z_old = Z(j);
        H(j) = X(j,:)*S(:,j);
        W(j) = H(j)/(Lam(j)-H(j));
        m = X(j,:)*B;
        m = m - W(j)*(Z(j)-m);
        q = Lam(j)*(W(j)+1);

        % Draw Z(j) from truncated Normal
        Z(j) = sampleNormalInd(m,q,y(j));

        % Update B
        B = B + ((Z(j)-z_old)/Lam(j))*S(:,j);
    end

    % Draw new Value of Beta

    T = mvnrnd(zeros(p,1),eye(p))';
    beta(:,i) = B + L*T;

    % Draw new value of mixing variances
    for j = 1:n
        R = Z(j)-X(j,:)*beta(:,i);
        Lam(j) = sampleLambda(abs(R));
    end
end
end % function


%% scheme 2
function [beta] = logist2Sample2(X,y,v,numSamples,verbose)

[n,p] = size(X);

% Initialize Mixing Weights (we use a vector rather than matrix in paper)
Lam = ones(n,1);

% Draw initial Z from truncated Normal
Z = abs(randraw('logistic',[0 1],n)).*sign(y-.5);

for i = 1:numSamples
  if verbose, fprintf('Drawing sample %d\n',i); end
  
  % v and V should be psd, so no need to use slow pinv
  V = (X'*diag(Lam.^-1)*X + v^-1)^-1;
  L = chol(V)';
  
  B = V*X'*diag(Lam.^-1)*Z;
  T = mvnrnd(zeros(p,1),eye(p))';
  beta(:,i) = B + L*T;
  
  % Update {Z,Lam}
  for j = 1:n
    m = X(j,:)*beta(:,i);
    
    % draw Z(j) from truncated logistic
    Z(j) = sampleLogisticInd(m,1,y(j));
    
    % draw new value for mixing variance
    R = Z(j)-m;
    Lam(j) = sampleLambda(abs(R));
  end
end
end

%% sampleNormalInd
function [sample] = sampleNormalInd(m,v,y)
s = sign(y-.5);
U=rand;
sample = logisticInversecdf(U,m,v);
while sign(sample)~=s
    if s > 0
        U = 1-rand*(1-U);
        if U == 1
            fprintf('Numerically Unstable\n');
            pause;
            break;
        end
        sample = logisticInversecdf(U,m,v);
    else
        U = rand*U;
        if U == 0
            fprintf('Numerically Unstable\n');
            break;
        end
        sample = logisticInversecdf(U,m,v);
    end
end
end

function [F] = logisticInversecdf(p,a,b)
F = a + b*(log(p)-log(1-p));
end

%% sampleLambda
function [lambda] = sampleLambda(r)
ok = 0;
while ~ok
    Y = randn;
    Y = Y*Y;
    Y = 1+(Y-sqrt(Y*(4*r+Y)))/(2*r);
    U = rand;

    if U <= 1/(1+Y)
        lambda = r/Y;
    else
        lambda = r*Y;
    end

    % Now, lambda ~ GIG(0.5,1,r^2)
    U = rand;

    if U > 4/3
        ok = rightmost_interval(U,lambda);
    else
        ok = leftmost_interval(U,lambda);
    end
end
end

function [OK] = rightmost_interval(U,lambda)
Z = 1;
X = exp(-.5*lambda);
j = 0;

while 1
    j = j + 1;
    Z = Z-((j+1)^2)*X^(((j+1)^2) - 1);

    if Z > U
        OK = 1;
        return;
    end

    j = j + 1;
    Z = Z+((j+1)^2)*X^(((j+1)^2) - 1);

    if Z < U
        OK = 0;
        return;
    end

end
end

function [OK] = leftmost_interval(U,lambda)
H = 0.5*log(2) + 2.5*log(pi) - 2.5*log(lambda) - (pi^2)/(2*lambda) + 0.5*lambda;
lU = log(U);
Z = 1;
X = exp((-pi^2)/(2*lambda));
K = lambda/pi^2;
j = 0;
while 1
    j = j + 1;
    Z = Z - K*X^((j^2)-1);

    if H + log(Z) > lU
        OK = 1;
        return;
    end

    j = j + 1;
    Z = Z + ((j+1)^2)*X^(((j+1)^2) -1);

    if H + log(Z) < lU
        OK = 0;
        return;
    end
end
end

