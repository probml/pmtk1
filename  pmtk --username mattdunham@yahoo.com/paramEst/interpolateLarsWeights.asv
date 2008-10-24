function Wbig = interpolateLarsWeights(Wfull,lambdas,X,y)
% Wbig(i,j) = w(j) using lambdas(i) for the L1 penalty in lasso
% Input:
% Wfull is the output of lars; each row is a solution (gets denser)
% lambdas - desired range
% X: input data, each row is a case
% y: input data

% Written by Matthew Dunham

%We have the values of the weights at each 'critical point' where
%weights changes sign from lars. Since the weights w.r.t. lambda
%are piecewise linear, we can just perform linear interpolation to
%get the weights corresponding to lambdas between these points.

Wfull = Wfull(end:-1:1,:); %reverse order for interp1q, (now least regularized to most)
criticalPoints = recoverLambda(X,y,Wfull)'; %in ascending order of magnitude.
tooBig = lambda > criticalPoints(end);%can't interpolate outside of the range of criticalPoints
Winterp = interp1q(criticalPoints,Wfull,lambdas(~tooBig)');
Wbig = [Winterp; zeros(sum(tooBig), d)]; % since, if lambda > lambda_max, all weights 0.


end
