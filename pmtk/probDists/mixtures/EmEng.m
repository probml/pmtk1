classdef EmEng  < OptimEng 
  % Generic expectation maximization
  % It implements multiple restarts
  % We should add support for mini-batches and other speedups

 methods(Abstract)
     %displayProgress(eng,data,currentLL,iter,r);
     eng = initializeEM(eng, model,data,r);
     [ess, currentLL] = Estep(eng, data);
     eng = Mstep(eng, ess);
 end
   
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
        '-verbose', false, ...
        '-nrestarts' ,1, ...
        '-maxIter'   ,50    ,...
        '-convTol'    ,1e-3);
    end
    
    function [model, LL, niter] = fit(eng, model, data)
      if(~isempty(model.transformer))
        [data, model.transformer] = train(model.transformer,data);
      end
      bestLL = -inf;
      niter = 0;
      for r = 1:eng.nrestarts,
        [model, currentLL, iter] = fitNoRestarts(eng, model, data, r);
        if(currentLL > bestLL)
          bestModel = model;
          bestLL = currentLL;
        end
        niter = niter + iter; % sum up over all restarts 
      end % for r
      model = bestModel;
      LL = bestLL;
    end % fit
    
    function [model, currentLL, niter] = fitNoRestarts(eng, model, data, r)
      niter = 0;
      model = initializeEM(eng,model,data,r);
      converged = false; iter = 0;
      currentLL = -inf; % sum(logprob(model,data)) + logprior(model);
      while(not(converged))
        prevLL = currentLL;
        [ess, currentLL] = Estep(eng,model, data);
        model = Mstep(eng, model, ess);
        iter = iter + 1;
        if (eng.verbose), displayProgress(eng, model,data,currentLL,iter,r);end
        converged = iter >= eng.maxIter || convergenceTest(currentLL, prevLL, eng.convTol);
        if currentLL < prevLL
          warning('EmEng:fit', 'EM not monotonically increasing objective')
        end
      end
    end % fitNoRestarts
    
    function displayProgress(eng, model,data,loglik,iter,r) %#ok that ignores model
      % override in subclass with more informative display
      t = sprintf('EM restart %d iter %d, negloglik %g\n',r,iter,-loglik);
      fprintf(t);
    end
    
  end % methods
  
end % class
