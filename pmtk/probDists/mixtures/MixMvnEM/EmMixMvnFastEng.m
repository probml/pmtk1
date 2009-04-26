classdef EmMixMvnFastEng  < OptimEng 
  % expectation maximization for mixtures of gaussians
 % This class bypasses the EmEng and EmMixEng for speed,
 % and calls EMforGMM, which hard-codes the relevant computations.

  properties
    verbose;
    plot;
    nrestarts;
    convTol;
    maxIter;
  end

  methods
    function eng = EmMixMvnFastEng(varargin)
      [eng.verbose,  eng.plot, eng.nrestarts, eng.maxIter, eng.convTol] = processArgs(varargin,...
        '-verbose', false, ...
        '-plot', false, ...
        '-nrestarts' ,1, ...
        '-maxIter'   ,50    ,...
        '-convTol'    ,1e-3);
    end

    function [model, loglikTrace, itr] = fit(eng, model, X)
      [model.distributions, model.mixingDistrib, loglikTrace, itr] = EMforGMM(model.distributions, ...
        model.mixingDistrib, X, '-verbose', eng.verbose, ...
        '-maxItr', eng.maxIter, '-tol', eng.convTol, '-nrestarts', eng.nrestarts);
    end % fit
   
  end % methods



end % class
