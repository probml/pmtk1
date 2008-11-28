classdef McmcDist  < SampleDist
   % Markov Chain Monte Carlo

   properties
     acceptRatio;
   end
   
  methods
    function obj = McmcDist(varargin)
      [obj.samples, naccept] = mcmcSample(varargin{:});
      Nsamples = size(obj.samples,1);
      obj.acceptRatio = naccept/Nsamples;
    end
    
  end
    
end