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
    [U,D,V] = svd(XtrainC,'econ');
    D2 = diag(D.^2);
    if lambda==0
      w = pinv(XtrainC)*ytrainC;
    else
      w  = V*diag(1./(D2 + lambda))*D*U'*ytrainC;
    end
  otherwise
    error(['unknown method ' method])
end

if computeOffset
  w0 = ybar - xbar*w;
  w = [w0; w];
end
