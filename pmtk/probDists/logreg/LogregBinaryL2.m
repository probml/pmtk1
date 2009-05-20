classdef LogregBinaryL2 < LogregBinary
%% Binary logistic regression with L2 prior on weights

    properties
      lambda;
    end  
    
    %% Main methods
    methods

     function m = LogregBinaryL2(varargin)
      % LogregBinaryL2(lambda, transformer, verbose,  w, w0, optMethod,
      % labelSpace)
      % optMethod can be any minFunc method (default lbfgs)
      % or 'irls' (Newton) or 'sgd' (stochastic gradient descent)
      % or 'perceptron'
      [m.lambda, m.transformer,  m.verbose,  m.w, m.w0,  m.optMethod, ...
        m.labelSpace, m.addOffset] = ...
        processArgs( varargin ,...
        '-lambda', [], ...
        '-transformer', [], ...
        '-verbose', false, ...
        '-w'          , [], ...
        '-w0', [], ...
        '-optMethod', 'newton', ... % lbfgs gives problems
        '-labelSpace', [], ...
        '-addOffset', true);
     end
    
     
     function [model,output] = fit(model,D)
       % m = fit(m, D) Compute MAP estimate
       % D is DataTable containing:
       % X(i,:) is i'th input; do *not* include a column of 1s
       % y(i) is i'th response
       X = D.X; y = D.Y;
       if ~isempty(model.transformer)
         [X, model.transformer] = train(model.transformer, X);
         if addOffset(model.transformer), error('don''t add column of 1s'); end
       end
       n = size(X,1);
       if model.addOffset
         X = [ones(n,1) X];
       end
       [n,d] = size(X);
       U = unique(y);
       if isempty(model.labelSpace), model.labelSpace = U; end
       y12 = canonizeLabels(y, model.labelSpace); %1,2
       y01 = y12-1; % 0,1
       ypm1 = 2*y01 - 1; % -1,+1
       winit = zeros(d,1);
       switch model.optMethod
         case 'irls',
           [w, output] = LogregBinaryL2.logregFitNewton(X, y01, model.lambda, model.addOffset);
         case 'sgd',
           [w, output] = LogregBinaryL2.logregSGD(X, y01, model.lambda);
         case 'perceptron',
           [w, output] = LogregBinaryL2.perceptron(X, ypm1, model.lambda);
         otherwise
           objective = @(w,junk) LogregBinaryL2.logregNLLgradHess(w, X, y01, model.lambda, model.addOffset);
           options.Method = model.optMethod;
           options.Display = model.verbose;
           [w, f, exitflag, output] = minFunc(objective, winit, options);
       end
       if model.addOffset
         model.w0 = w(1);
         model.w = w(2:end);
       else
         model.w0 = 0;
         model.w = w;
       end
     end
      

    end % methods

    methods(Static = true)
      function [f,g,H] = logregNLLgradHess(beta, X, y, lambda, offsetAdded)
        % gradient and hessian of negative log likelihood for logistic regression
        % beta should be column vector
        % Rows of X contain data
        % y(i) = 0 or 1
        % lambda is optional strength of L2 regularizer
        % set offsetAdded if 1st column of X is all 1s
        if nargin < 4,lambda = 0; end
        check01(y);
        mu = 1 ./ (1 + exp(-X*beta)); % mu(i) = prob(y(i)=1|X(i,:))
        if offsetAdded, beta(1) = 0; end % don't penalize first element
        f = -sum( (y.*log(mu+eps) + (1-y).*log(1-mu+eps))) + lambda/2*sum(beta.^2);
        %f = f./nexamples;
        g = []; H  = [];
        if nargout > 1
          g = X'*(mu-y) + lambda*beta;
          %g = g./nexamples;
        end     
        if nargout > 2
          W = diag(mu .* (1-mu)); %  weight matrix
          H = X'*W*X + lambda*eye(length(beta));
          %H = H./nexamples;
        end
      end 
      
      function [beta, C] = logregFitNewton(X, y, lambda, offsetAdded)
        % Iteratively reweighted least squares for logistic regression
        %
        % Rows of X contain data
        % y(i) = 0 or 1
        % lambda is optional strenght of L2 regularizer
        %
        % Returns beta, a row vector
        % and C, the asymptotic covariance matrix
        
        %#author David Martin
        %#modified Kevin Murphy
        
        if nargin < 3, lambda = 0; end
        [N,p] = size(X);
        beta = zeros(p,1); % initial guess for beta: all zeros
        iter = 0;
        tol = 1e-6; % termination criterion based on loglik
        nll = 0;
        maxIter = 1000;
        done = false;
        while ~done
          iter = iter + 1;
          nll_prev = nll;
          [nll, g, H] = LogregBinaryL2.logregNLLgradHess(beta, X, y, lambda, offsetAdded);
          beta = beta - H\g; % more stable than beta - inv(hess)*deriv
          if abs((nll-nll_prev)/(nll+nll_prev)) < tol, done = true; end;
          nllTrace(iter) = iter;
          if iter > maxIter
            warning('did not converge');
            done = true;
          end
        end;
        [nll, g, H] = LogregBinaryL2.logregNLLgradHess(beta, X, y, lambda, offsetAdded); 
        C = inv(H);
        output.C = C;
        output.nllTrace = nll;
      end
      
      function [w, output] = logregSGD(X, y, lambda, varargin)
        % X(i,:) is i'th case
        % y(i) = 0 or 1
        % lambda l2 regularizer
        % optional arguments as in minFunc - MaxIter and TolX
        %
        % output currently only contains ftrace
        [maxIter, tolX] = process_options(varargin, ...
          'maxIter', 50, 'tolX', 1e-4);
        check01(y);
        [n d] = size(X);
        w = rand(d,1);
        wold = w;
        ftrace = [];
        eta = 1; % step size
        for iter=1:maxIter
          for i=1:n
            xi = X(i,:)';
            mui = sigmoid(w'*xi);
            gi = (y(i)-mui)*xi - (lambda/n)*w;
            w = w + eta*gi;
            incr = norm(wold -w)/norm(wold);
            if (incr < tolX)
              break;
            end
            wold = w;
          end
          eta = 1/(iter+1); % step size decay
          if nargout > 1
            f = LogregBinaryL2.logregNLLgradHess(w, X, y, lambda);
            ftrace(iter) = f;
          end
        end
        output.ftrace = ftrace;
      end
      
      function [w, output] = perceptron(X, y,lambda)
        % X(i,:) is i'th case, y(i) = -1 or +1
        % lambda l2 regularizer
        % Based on code by Thomas Hoffman 
        [n d] = size(X);
        checkPM1(y);
        w = zeros(d,1);
        max_iter = 1000;
        nerrors = [];
        for iter=1:max_iter
          nerrors(iter) = 0;
          for i=1:n
            xi = X(i,:)';
            yhati = sign(w'*xi);
            if ( y(i)*yhati <= 0 ) % made an error
              w = w + y(i) * xi - (lambda/n)*w;
              nerrors(iter) = nerrors(iter) + 1;
            end
          end
          %fprintf('Iteration %d, errors = %d\n', iter, errors);
          if (nerrors(iter)==0)
            break;
          end
        end
        output.nerrorsTrace = nerrors;
      end

    end% methods
    
end
