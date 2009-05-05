classdef MhInfEng  < InfEng 
   % Metropolis Hastings
   % Model must support the following methods
 % lp = logprob(model, x, false) % unnormalized log posterior
 %  xinit = mcmcInitSample(model, visVars, visVals);
 
   properties
     Nsamples; Nburnin; thin;
     proposal;
     symmetric;
     Nchains;
     convDiag;
     verbose;
     samples;
   end
   
  methods
    function obj = MhInfEng(varargin)
      [obj.Nsamples, obj.Nburnin, obj.thin, obj.proposal, obj.symmetric, ...
        obj.Nchains, obj.verbose] = ...
        process_options(varargin, 'Nsamples', 500, 'Nburnin', 100, ...
        'thin', 1, 'proposal', [], 'symmetric', true, ...
        'Nchains', 3, 'verbose', false);
      obj.samples = [];
    end
   
     
     function [eng, logZ, other] = condition(eng, model, visVars, visVals)
       V = lookupIndices(visVars, model.domain);
       %function x = addVisData(x, V, visVars)
       %  x(V) = visVars;
       %end
       %targetFn = @(xh) logprobUnnormalized(model, MhInfEng.addVisData(xh, V, visVals));
       targetFn = @(xh) logprob(model, MhInfEng.addVisData(xh, V, visVals), false);
       %targetFn = @(xh) logprobUnnormalized(model, xh);
       %hidVars = setdiffPMTK(model.domain, visVars);
       %targetFn = @(xh) logprobUnnormalized(model, xh, 'domain', hidVars, ...
       %  'visVars', visVars, 'visVals', visVals);
      xinit = mcmcInitSample(model, visVars, visVals);
      ndims = length(xinit);
      samples = zeros(eng.Nsamples, ndims, eng.Nchains);
      for c=1:eng.Nchains
        if eng.verbose
          fprintf('MH: starting to collect %d samples from chain %d of %d\n', ...
            eng.Nsamples, c, eng.Nchains);
        end
        if c>1, xinit = mcmcInitSample(model, visVars, visVals); end
        [samples(:,:,c)] = ...
          mhSample('symmetric', eng.symmetric, 'target', targetFn, 'xinit', xinit, ...
         'Nsamples', eng.Nsamples, 'Nburnin', eng.Nburnin, 'thin', eng.thin, ...
         'proposal',  eng.proposal);
      end
      if eng.Nchains > 1
        [eng.convDiag.Rhat, eng.convDiag.converged] = epsrMultidim(samples);
        samples = permute(samples, [1 3 2]); % s,c,j
        samples = reshape(samples, eng.Nsamples*eng.Nchains, ndims); % s*c by j
      end
       % The samples only contain values of the hidden variables, not all
       % the variables, so we need to 'label' the columns with the right
       % domain
       hidVars = setdiffPMTK(model.domain, visVars);
       eng.samples = SampleBasedDist(samples, hidVars);
       logZ = [];
       other = eng.convDiag;
     end
    
     function [postQuery,eng] = marginal(eng, queryVars)
       if isempty(eng.samples), error('must first call condition'); end
       postQuery = marginal(eng.samples, queryVars);
     end

     function [samples] = sample(eng, n)
       if isempty(eng.samples), error('must first call condition'); end
       samples = sample(eng.samples, n);
     end

  end
  
  %% Static methods
  methods(Static = true)
     function x = addVisData(x, V, visVars)
       x(V) = visVars;
     end
  end
  
end
