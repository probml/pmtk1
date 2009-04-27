classdef LinregL2 < Linreg
%% Ridge Regression  (Single Variate Output)


    properties
        lambda;
        method;
    end
 
  
    %% Main methods
    methods
      function obj = LinregL2(varargin)
        % m = LinregL2(lambda, transfomer, w, sigma2, method, dof)
        % method is one of {'ridgeQR', 'ridgeSVD'}
        [obj.lambda, obj.transformer, obj.w, obj.sigma2, obj.method, obj.df] = ...
          processArgs(varargin,...
          '-lambda'      , 0, ...
          '-transformer', []                      ,...
          '-w'          , []                      ,...
          '-sigma2'     , []                      , ....
          '-method', 'ridgeQR', ...
          '-df', 0);
      end
       
        function model = fit(model,varargin)
          % m = fit(m, X, y)
          % X(i,:) is i'th input; do *not* include a column of 1s
          % y(i) is i'th response
          [X, y] = processArgs(varargin, '-X', [], '-y', []);
          if ~isempty(model.transformer)
            [X, model.transformer] = train(model.transformer, X);
          end
          onesAdded = ~isempty(model.transformer) && addOffset(model.transformer);
          
          if onesAdded
            X = X(:,2:end); % remove leading column of 1s
          end
          model.w = ridgereg(X, y, model.lambda, model.method, onesAdded);
          model.df = dofRidge(model, X, model.lambda);
          
          n = size(X,1);
          if onesAdded
            X = [ones(n,1) X]; % column of 1s for w0 term
          end
          yhat = X*model.w;
          model.sigma2 = mean((yhat-y).^2);
          model.ndimsX = size(X,2);
          model.ndimsY = size(y,2);
        end

    end % methods


  
end % class

%%%%%%%%%%
function [w]= ridgereg(X, y, lambda, method, computeOffset)

if computeOffset
  % center input and output, so we can estimate w0 separately
  % since we don't want to shrink w0
  xbar = mean(X);
  XC = X - repmat(xbar,size(X,1),1);
  y = y(:);
  ybar = mean(y);
  yC = y-ybar;
else
  XC = X;
  yC = y;
end

switch lower(method)
  case 'ridgeqr'
    if isscalar(lambda)
      if lambda==0
        w = XC \ yC; % least squares
      else
        d = size(XC,2);
        XX  = [XC; sqrt(lambda)*eye(d)];
        yy = [yC; zeros(d,1)];
        w  = XX \ yy; % ridge
      end
    else
      XX  = [XC; lambda];
      yy = [yC; zeros(size(lambda,1),1)];
      w  = XX \ yy; % generalized ridge
    end
  case 'ridgesvd'
    [U,D,V] = svd(XC,'econ');
    D2 = diag(D.^2);
    if lambda==0
      w = pinv(XC)*yC;
    else
      w  = V*diag(1./(D2 + lambda))*D*U'*yC;
    end
  otherwise
    error(['unknown method ' method])
end

if computeOffset
  w0 = ybar - xbar*w;
  w = [w0; w];
end
end

%%%%%%%%%%
function df = dofRidge(model, X, lambdas)
% Compute the degrees of freedom for a given lambda value
% Elements of Statistical Learning p63
if ~isempty(model.transformer)
  X = train(model.transformer, X);
  if addOffset(model.transformer)
    X = X(:,2:end);
  end
end
xbar = mean(X);
XC = X - repmat(xbar,size(X,1),1);
[U,D,V] = svd(XC,'econ');                                           %#ok
D2 = diag(D.^2);
nlambdas = length(lambdas);
df = zeros(nlambdas,1);
for i=1:nlambdas
  df(i) = sum(D2./(D2+lambdas(i)));
end
end