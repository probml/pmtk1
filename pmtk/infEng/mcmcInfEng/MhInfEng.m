classdef McmcInfEng  < InfEng 
   % Markov Chain Monte Carlo

   properties
     acceptRatio;
     samples;
     method; Nsamples; Nburnin; thin;
   end
   
  methods
    function obj = McmcInfEng(varargin)
      [method, fullCond, target, proposal, Nsamples, Nburnin, thin, xinit] = ...
        process_options(varargin, 'method', [], 'fullCond', [], 'target', [], ...
        'proposal', [], 'Nsamples', 1000, 'Nburnin', 100, 'thin', 1);
      obj.method = method;
    end
    
    function obj = run(obj, xinit, fullCond)
      % starts running as soon as you create it...
      [obj.samples, naccept] = mcmcSample(varargin{:});
      Nsamples = size(obj.samples,1);
      obj.acceptRatio = naccept/Nsamples;
    end
    
  end
    
end