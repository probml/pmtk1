classdef GibbsInfEng  < InfEng 
   % Gibbs sampling
   % Model must support following methods
% fullCond = makeFullConditionals(model, visVars, visVals);
%  xinit = mcmcInitSample(model, visVars, visVals);
      
   properties
     Nsamples; Nburnin; thin;
     Nchains;
     convDiag;
     verbose;
     samples;
   end
   
  methods
    function obj = GibbsInfEng(varargin)
      [obj.Nsamples, obj.Nburnin, obj.thin, obj.Nchains, obj.verbose] = ...
        process_options(varargin, 'Nsamples', 500, 'Nburnin', 100, ...
        'thin', 1, 'Nchains', 3, 'verbose', false);
    end
   
    function [eng] = condition(eng, model, visVars, visVals)
      fullCond = makeFullConditionals(model, visVars, visVals);
      xinit = mcmcInitSample(model, visVars, visVals);
      ndims = length(xinit);
      samples = zeros(eng.Nsamples, ndims, eng.Nchains);
      for c=1:eng.Nchains
        if eng.verbose
          fprintf('starting to collect %d samples from chain %d of %d\n', ...
            eng.Nsamples, c, eng.Nchains);
        end
        if c>1, xinit = mcmcInitSample(model, visVars, visVals); end
        [samples(:,:,c)] = gibbsSample(fullCond, xinit, eng.Nsamples, eng.Nburnin, eng.thin);
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
       eng.samples = SampleDist(samples, hidVars); % , model.support(hidVars));
    end
    
    
     function [postQuery,eng] = marginal(eng, queryVars)
       if isempty(eng.samples), error('must first call condition'); end
       postQuery = marginal(eng.samples, queryVars);
     end
    
     function [samples] = sample(eng, n)
       if isempty(eng.samples), error('must first call condition'); end
      samples = sample(eng.samples,n);
    end
  
    
  end
    
end