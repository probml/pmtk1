classdef NoPrior < ProbDist
% Used to represent absence of prior to force MAP estimation to work like MLE
% We use an object rather than a string (such as 'none')
% so we can write
%   switch class(prior)
%     case NoPrior, ...
%     case XXX, ...
%   end

    methods
      function m = NoPrior()
      end
      
      function p = logprob(m, X)
        n = size(X,1);
        p = zeros(n,1);
      end
    end
       
end