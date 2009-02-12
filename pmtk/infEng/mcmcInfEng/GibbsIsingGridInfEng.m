classdef GibbsIsingGridInfEng  < InfEng 
   % Gibbs sampling for an Ising grid
      
   properties
     Nsamples; Nburnin; thin;
     Nchains;
     convDiag;
     verbose;
     samples;
   end
   
  methods
    function obj = GibbsIsingGridInfEng(varargin)
      [obj.Nsamples, obj.Nburnin, obj.thin, obj.Nchains, obj.verbose] = ...
        process_options(varargin, 'Nsamples', 500, 'Nburnin', 100, ...
        'thin', 1, 'Nchains', 3, 'verbose', false);
    end
   
    function avgX = postMean(eng, model, fn, visVars, visVals)
      % If visVars=visVals=[], we will sample from the prior
      % Otherwise, we assume visVars='visible'
      % so visVals should be an n*m matrix
      [M,N] = size(visVals);
      Npixels = M*N;
      localEvidence = zeros(Npixels, 2);
      for k=1:2
        localEvidence(:,k) = exp(logpdf(model.CPD{k}(y(:))));
      end
      Xinit = GibbsIsingGridInfEng.computeInit(localEvidence);
      avgX = GibbsIsingGridInfEng.computeMean(J, M, N, localEvidence, Xinit, ...
        eng.Nsamples, eng.Nburnin, eng.thin);
    end
    
    
  end
   
  methods(Static=true)
    
    function X = computeInit(localEvidence)
      [junk, guess] = max(localEvidence, [], 2);  % start with best local guess
      X = ones(M, N);
      offState = 1; onState = 2;
      X((guess==offState)) = -1;
      X((guess==onState)) = +1;
    end
      
    function avgX = computeMean(J, M, N, localEvidence, Xinit, Nsamples, burnIn, thin);
      X = Xinit;
      avgX = zeros(size(X));
      offState = 1; onState = 2;
      for iter =1:Nsamples
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
        if iter > burnIn
          avgX = avgX+X;
        end
      end
      nSamples = (maxIter-burnIn);
      avgX = avgX/nSamples;
    end

  end % static

end