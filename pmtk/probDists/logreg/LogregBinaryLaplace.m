classdef LogregBinaryLaplace < ProbDist 
%% Binary logistic regression with Gaussian prior
% We use Laplace approximation to the posterior

properties
  wDist;
  transformer;            % A data transformer object, e.g. KernelTransfor
  labelSpace;           % The suppport of the target y, e.g. [0,1], [-1,+1],
  lambda; % precision of diagonal Gaussian
  verbose;
  optMethod;
  predMethod;
  nsamples;
end



    
    %% Main methods
    methods

       function m = LogregBinaryLaplace(varargin)
       % LogregBinary(transformer, verbose, w, w0, optMethod,
      % labelSpace, predMethod, nsamples)
      % The optimizer is used to find the mode
      
      [m.lambda, m.transformer, m.verbose,   m.optMethod, m.labelSpace, ...
        m.predMethod, m.nsamples] = ...
        processArgs( varargin ,...
        '-lambda', [], ...
        '-transformer', [], ...
        '-verbose', false, ...
        '-optMethod', 'lbfgs', ...
        '-labelSpace', [], ...
        '-predMethod', 'sigmoidTimesGauss', ...
        '-nsamples', 100);
       end
        
       function wDist = getParamPost(model) % Mvn
         wDist = model.wDist;
       end
       
       function [model,output] = fit(model,D)
         % m = fit(m, D) Compute posterior estimate
         % D is DataTable containing:
         % X(i,:) is i'th input; do *not* include a column of 1s
         % y(i) is i'th response
         
         % First find mode
         tmp = LogregBinaryL2('-lambda', model.lambda, ...
           '-labelSpace', model.labelSpace, '-optMethod', model.optMethod, ...
           '-transformer', model.transformer);
         [tmp, output] = fit(tmp, D);
         wMAP = [tmp.w0; tmp.w];
         
         % Now compute Hessian
         X = D.X; y = D.Y;
         if ~isempty(model.transformer)
           [X, model.transformer] = train(model.transformer, X);
           if addOffset(model.transformer), error('don''t add column of 1s'); end
         end
         n = size(X,1);
         X = [ones(n,1) X];
         [n,d] = size(X);
         offsetAdded = true;
         U = unique(y);
         if isempty(model.labelSpace), model.labelSpace = U; end
         y12 = canonizeLabels(y, model.labelSpace); %1,2
         y01 = y12-1; % 0,1
         [nll,g,H] = LogregBinaryL2.logregNLLgradHess(wMAP, X, y01, model.lambda, offsetAdded);
         C = inv(H); %H = hessian of neg log lik
         model.wDist = MvnDist(wMAP, C);
       end
       
       
       function [yhat, pred] = predict(model,D)
         % yhat(i) = most probable label for X(i,:)
         % pred(i) = p(y|X(i,:), w) a BernoulliDist
         X = D.X;
         if ~isempty(model.transformer)
           X = test(model.transformer, X);
         end
         [n,d] = size(X);
         X = [ones(n,1) X];
         switch model.predMethod
           case 'sigmoidTimesGauss'
             p = LogregBinaryLaplace.sigmoidTimesGauss(X, model.wDist.mu(:), model.wDist.Sigma);
             pred = BernoulliDist('-mu', p, '-support', model.labelSpace);
           case 'mc'
             ns = model.nsamples;
             wsamples = sample(model.wDist,ns);
             psamples = zeros(n,ns);
             for s=1:ns
               psamples(:,s) = sigmoid(X*wsamples(s,:)');
             end
             pred = SampleBasedDist(psamples');
             p = mean(pred);
         end
         yhat = ones(n,1);
         ndx2 = (p > 0.5);
         yhat(ndx2) = 2;
         yhat = model.labelSpace(yhat);
       end
    
     function p = logprob(obj, D)
      % p(i) = log p(y(i) | D.X(i,:), obj.w), D.y(i) in 1...C
       X = D.X; y = D.Y; 
       y = canonizeLabels(y, obj.labelSpace);
       obj.predMethod = 'sigmoidTimesGauss'; % get table out
      [yhat, pred] = predict(obj,D);
      p = pmf(pred)'; % n by 1
      %n = size(X,1); 
      %X = [ones(n,1) X];
      %p = LogregBinaryLaplace.sigmoidTimesGauss(X, model.wDist.mu(:), model.wDist.Sigma);
      %P = [p 1-p];       
      Y = oneOfK(y, 2); %obj.nclasses);
      %p =  sum(sum(Y.*log(P)));
      P = [p(:) 1-p(:)];
      p =  sum(Y.*log(P), 2);
    end
     
    end % methods
    
    methods(Static = true)
      function p = sigmoidTimesGauss(X, wMAP, C)
        % Bishop'06 p219
        mu = X*wMAP;
        n = size(X,1);
        if n < 1000
          sigma2 = diag(X * C * X');
        else
          % to save memory, use non-vectorized version
          sigma2 = zeros(1,n);
          for i=1:n
            sigma2(i) = X(i,:)*C*X(i,:)';
          end
        end
        kappa = 1./sqrt(1 + pi.*sigma2./8);
        p = sigmoid(kappa .* reshape(mu,size(kappa)));
      end
      
    end % static
end

   