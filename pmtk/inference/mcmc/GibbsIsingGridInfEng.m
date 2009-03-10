classdef GibbsIsingGridInfEng  < InfEng 
   % Gibbs sampling for an Ising grid
      
   properties
     Nsamples; Nburnin; thin;
     Nchains;
     convDiag;
     verbose;
     samples;
     progressFn;
   end
   
  methods
    function obj = GibbsIsingGridInfEng(varargin)
      [obj.Nsamples, obj.Nburnin, obj.thin, obj.Nchains, ...
        obj.verbose, obj.progressFn] = ...
        process_options(varargin, 'Nsamples', 500, 'Nburnin', 100, ...
        'thin', 1, 'Nchains', 1, 'verbose', false, 'progressFn', []);
    end
   
    function avgX = postMean(eng, model, visVars, visVals)
      % visVars will be ignored
      % visVals should be an n*m matrix
      [M,N] = size(visVals);
      Npixels = M*N;
      localEvidence = zeros(Npixels, 2);
      for k=1:2
        localEvidence(:,k) = exp(logprob(model.CPDs{k}, visVals(:)));
      end
      Xinit = GibbsIsingGridInfEng.computeInit(localEvidence, M, N);
      avgX = GibbsIsingGridInfEng.computeMean(model.J, M, N, localEvidence, Xinit, ...
        eng.Nsamples, eng.Nburnin, eng.thin, eng.progressFn);
    end
    
    function [postQuery,eng] = marginal (eng, queryVars)  
      error('not supported')
    end
    
    function  eng = condition(eng, model, visVars, visVals) 
      error('not supported')
    end
    
    function samples = sample(eng, n)                   
      error('not yet implemented')
    end
    
  end
   
  methods(Static=true)
    
    function X = computeInit(localEvidence, M, N)
      [junk, guess] = max(localEvidence, [], 2);  % start with best local guess
      X = ones(M, N);
      offState = 1; onState = 2;
      X((guess==offState)) = -1;
      X((guess==onState)) = +1;
    end
      
    function avgX = computeMean(J, M, N, localEvidence, Xinit, Nsamples, Nburnin, ...
        thin, progressFn)
      X = Xinit;
      avgX = zeros(size(X));
      S = (Nsamples + Nburnin);
      offState = 1; onState = 2;
      for iter =1:S
        % select a pixel at random
        ix = ceil( N * rand(1) ); iy = ceil( M * rand(1) );
        pos = iy + M*(ix-1);
        neighborhood = pos + [-1,1,-M,M];
        neighborhood(([iy==1,iy==M,ix==1,ix==N])) = [];
        % compute local conditional
        wi = sum( X(neighborhood) );
        p1  = exp(J*wi) * localEvidence(pos,onState);
        p0  = exp(-J*wi) * localEvidence(pos,offState);
        prob = p1/(p0+p1);
        if rand < prob
          X(pos) = +1;
        else
          X(pos) = -1;
        end
        if (iter > Nburnin) %&& (mod(iter, thin)==0)
          avgX = avgX+X;
        end
        if ~isempty(progressFn)
          feval(progressFn, X, iter);
        end
      end
      avgX = avgX/Nsamples;
    end

  end % static

end