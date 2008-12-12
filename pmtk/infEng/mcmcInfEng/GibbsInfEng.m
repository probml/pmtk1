classdef GibbsInfEng  < InfEng 
   % Gibbs sampling
   % Model must support mcmcInitSample and mkFullConditionals

   properties
     % Parameters that control the sampler
     Nsamples; Nburnin; thin;
   end
   
  methods
    function obj = GibbsInfEng(varargin)
      [obj.Nsamples, obj.Nburnin, obj.thin] = ...
        process_options(varargin, 'Nsamples', 1000, 'Nburnin', 100, 'thin', 1);
    end
   
    function [samples] = sample(eng, model, n)
      fullCond = makeFullConditionals(model);
      xinit = mcmcInitSample(model);
      [samples] = gibbsSample(fullCond, xinit, n, eng.Nburnin, eng.thin);
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
       fc = makeFullConditionals(model, visVars, visVals);
       xinit = initMcmcSample(model, visVars, visVals);
       hidVars = mysetdiff(model.domain, visVars);
       [samples] = gibbsSample(fc, xinit, ...
         'Nsamples', n, 'Nburnin', eng.Nburnin, 'thin', eng.thin);
       % The samples only contain values of the hidden variables, not all
       % the variables, so we need to 'label' the columns with the right
       % domain
       postQuery = SampleDist(samples, hidVars);
       if (nargin >= 4) && ~isempty(queryVars)
         postQuery = marginal(postQuery, queryVars);
       end
     end
    
  end
    
end