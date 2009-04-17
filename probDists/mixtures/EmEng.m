classdef EmEng  < OptimEng 
  % Generic expectation maximization
  % It implements multiple restarts
  % We should add support for mini-batches and other speedups

%   Model must support the following methods
%
%   model = initializeEM(model,data,r);
%   [ess, currentLL] = Estep(model, data);
%   model = Mstep(model, ess);
%   displayProgress(model,data,currentLL,iter,r)

   properties
     verbose;
     nrestarts;
     convTol;
     maxIter;
   end
   
  methods
    function eng = EmEng(varargin)
      % EmEng(verbose, nrestarts, maxIter, convTol)
      [eng.verbose, eng.nrestarts, eng.maxIter, eng.convTol] = processArgs(varargin,...
        'verbose', true, ...
        'nrestarts' ,3, ...
        'maxIter'   ,50    ,...
        'convTol'    ,1e-3);
    end
    
    function [model, LL, niter] = fit(eng, model, data)
      if(~isempty(model.transformer))
        [data, model.transformer] = train(model.transformer,data);
      end
      bestLL = -inf;
      niter = 0;
      for r = 1:eng.nrestarts,
        model = initializeEM(model,data,r);
        converged = false; iter = 0;
        currentLL = -inf; % sum(logprob(model,data)) + logprior(model);
        while(not(converged))
          prevLL = currentLL;
          [ess, currentLL] = Estep(model, data);
          model = Mstep(model, ess);
          iter = iter + 1;
          if (eng.verbose), displayProgress(model,data,currentLL,iter,r);end
          converged = iter >= eng.maxIter || convergenceTest(currentLL, prevLL, eng.convTol);
          if currentLL < prevLL
            warning('EM not monotonically increasing objective')
          end
        end
        if(currentLL > bestLL)
          bestModel = model;
          bestLL = currentLL;
        end
        niter = niter + iter; % sum up over all restarts 
      end % for r
      model = bestModel;
      LL = bestLL;
    end % fit
    
  end % methods
  
end % class
