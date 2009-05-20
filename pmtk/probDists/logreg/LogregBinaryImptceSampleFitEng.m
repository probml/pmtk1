classdef LogregBinaryImptceSampleFitEng < OptimEng
 % IS with Gaussian proposal centered at MAP for
% Binary logistic regression with spherical Gaussian prior

properties
  nsamples;
  verbose;
end


%% Main methods
methods
  
  function m = LogregBinaryImptceSampleFitEng(varargin)
    % LogregBinaryImptceSampleFitEng(nsamples, verbose, scheme)
    [m.nsamples, m.verbose] = ...
      processArgs( varargin ,...
      '-nsamples', 1000, ...
      '-verbose', false);
  end
  
  function [model, out] = fit(eng, model, D)
    % m = fit(eng, m, D) Compute posterior estimate
    % D is DataTable containing:
    % X(i,:) is i'th input; do *not* include a column of 1s
    % y(i) is i'th response
    X = D.X;
   if ~isempty(model.transformer)
     [X, model.transformer] = train(model.transformer, X);
     if addOffset(model.transformer), error('don''t add column of 1s'); end
   end
   y01 = getLabels(D, '01');
   
   % First find mode
    tmp = LogregBinaryL2(model.lambda, model.transformer, ...
     '-addOffset', model.addOffset, '-verbose', false, '-optMethod', 'newton');
   tmp = fit(tmp, D);
   
   % Then find Hessian at mode
   if model.addOffset
     n = size(X,1);
     X = [ones(n,1) X];
     wMAP = [tmp.w0; tmp.w];
   else
     wMAP = tmp.w;
   end
   [nll2, g, H] = LogregBinaryL2.logregNLLgradHess(wMAP, X, y01, 0, model.addOffset);
   C = inv(H); %H = hessian of *neg* log lik
   
   % Now construct proposal
   d = length(wMAP);
   priorMu = zeros(d,1)';
   priorCov = (1/model.lambda)*eye(d);
   if  model.addOffset
     priorCov(1,1) = 1e5; % flat prior for offset
   end
   targetFn = @(w) -LogregBinaryL2.logregNLLgradHess(w, X, y01, 0, true) + ...
     log(mvnpdf(w(:)',priorMu,priorCov));
   
   samples = mvnrnd(wMAP, C, eng.nsamples)';
   % too tired to vectorize
   weights = zeros(1, eng.nsamples);
   for s=1:eng.nsamples
     weights(s) = targetFn(samples(:,s)) ./ (mvnpdf(samples(:,s)', wMAP(:)', C)+eps);
   end
   weights = normalize(weights);
   ndx = resamplingMultinomial(1:eng.nsamples, weights(:)); % no need for weights anymore
   model.paramDist.wsamples = samples(:,ndx);
   model.paramDist.weights = ones(1, eng.nsamples); %weights;
   out = [];
  end
  
  
end % methods


end
