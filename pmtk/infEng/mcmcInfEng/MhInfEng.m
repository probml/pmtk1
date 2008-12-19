classdef MhInfEng  < InfEng 
   % Metropolis Hastings
   % Model must support mcmcInitSample and logprob

   properties
     % Parameters that control the sampler
     Nsamples; Nburnin; thin;
     proposal;
     symmetric;
   end
   
  methods
    function obj = MhInfEng(varargin)
      [obj.Nsamples, obj.Nburnin, obj.thin, obj.proposal, obj.symmetric] = ...
        process_options(varargin, 'Nsamples', 1000, 'Nburnin', 100, ...
        'thin', 1, 'proposal', [], 'symmetric', true);
    end
   
    %eng = MhInfEng('targetFn', targetFn, 'proposalFn', proposalFn, 'initFn', initFn,...
    %            'Nsamples', 1000, 'Nburnin', 500);
    
    function [samples] = sample(eng, model, n)
      targetFn = @(x) logprob(model, x, false); % unnormalized
      xinit = mcmcInitSample(model);
      samples = mhSample('symmetric', eng.symmetric, 'target', targetFn, 'xinit', xinit, ...
        'Nsamples', n, 'Nburnin', eng.Nburnin, 'thin', eng.thin, 'proposal',  eng.proposal);
    end
    
     function [postQuery] = marginal(eng, model, queryVars)
       % Does a fresh sampling run every time.
       % If you want to compute multiple marginals,
       % pass in all at once as cell array. 
       [S] = sample(eng, model, eng.Nsamples);
       S = SampleDist(S, model.domain);
       if ~iscell(queryVars)
         postQuery = marginal(S, queryVars);
       else
         for i=1:length(queryVars)
           postQuery{i} = marginal(S, queryVars{i});
         end
       end
     end
    
     
     function [postQuery] = predict(eng, model, visVars, visVals, queryVars)
       V = lookupIndices(visVars, model.domain);
       targetFn = @(xh) logprob(model, MhInfEng.addVisData(xh, V, visVals),false); % unnormalized
       xinit = mcmcInitSample(model, visVars, visVals);
       samples = mhSample('symmetric', eng.symmetric, 'target', targetFn, 'xinit', xinit, ...
         'Nsamples', n, 'Nburnin', eng.Nburnin, 'thin', eng.thin, 'proposal',  eng.proposal);
       hidVars = mysetdiff(model.domain, visVars);
       postQuery = SampleDist(samples, hidVars);
       if (nargin >= 4) && ~isempty(queryVars)
         postQuery = marginal(postQuery, queryVars);
       end
     end

  end
  
  %% Private methods
  methods(Access = 'protected')
     function x = addVisData(x, V, visVars)
       x(V) = visVars;
     end
  end
  
end
