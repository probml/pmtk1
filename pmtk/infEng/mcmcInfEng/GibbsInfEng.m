classdef GibbsInfEng  < InfEng 
   % Gibbs sampling

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
      xinit = mvnrnd(model.mu, model.Sigma);
      [samples] = gibbsSample(fullCond, xinit, ...
        'Nsamples', n, 'Nburnin', eng.Nburnin, 'thin', eng.thin);
    end
    
     function [postQuery] = marginal(eng, model, Q)
       % If you want to compute multiple marginals,
       % avoid re-calling the sampler...
       [S] = sample(eng, model, eng.Nsamples);
       postQuery = marginal(SampleDist(S), Q);
     end
    
     function [postQuery] = predict(eng, model, visVars, visVals, Q)
       fc = makeFullConditionals(model, visVars, visVals);
       xinit = mvnrnd(model.mu, model.Sigma);
      [samples] = gibbsSample(fc, xinit, ...
        'Nsamples', n, 'Nburnin', eng.Nburnin, 'thin', eng.thin);
      postQuery = marginal(SampleDist(samples), Q);
     end
    
  end
    
end