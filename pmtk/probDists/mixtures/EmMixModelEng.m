classdef EmMixModelEng < EmEng
  
methods  
   function eng = EmMixModelEng(varargin)
      % EmMixModelEng(verbose, nrestarts, maxIter, convTol)
      [eng.verbose, eng.nrestarts, eng.maxIter, eng.convTol] = processArgs(varargin,...
        '-verbose', false, ...
        '-nrestarts' ,2, ...
        '-maxIter'   ,50    ,...
        '-convTol'    ,1e-3);
   end
    
    function [ess, L] = Estep(eng, model, data) %#ok skips eng
      %model = eng.model;
      
      [Rik, LL]  = inferLatent(model, data);
      N = size(data,1);
      L = sum(LL);
      L = L + logprior(model); % must add log prior for MAP estimation
      L = L/N;
      Rik = pmf(Rik)'; % convert from distribution to table of n*K numbers
      K = length(model.distributions);
      compSS = cell(1,K);
      for k=1:K
          compSS{k} = model.distributions{k}.mkSuffStat(data,Rik(:,k));
      end
      ess.compSS = compSS;
      ess.counts = colvec(normalize(sum(Rik,1)));
    end
  
    function [model, singular] = Mstep(eng, model, ess) %#ok skips eng
        K = length(model.distributions);
        for k=1:K
           
            
            
            model.distributions{k} = fit(model.distributions{k},'-suffStat',ess.compSS{k});
            [R, p] = chol(model.distributions{k}.Sigma);
            deter = det(model.distributions{k}.Sigma);
            singular =  ~isfinite(deter) || ~isfinite(p) || deter <=eps || ~(p==0);
             
            
            if(singular), return; end
        end
        mixSS.counts = ess.counts;
        model.mixingDistrib = fit(model.mixingDistrib,'-suffStat',mixSS);
    end
    
    function model = initializeEM(eng, model,data,r) %#ok that ignores data and r
      % override in subclass with more intelligent initialization
      K = length(model.distributions);
      %model = mkRndParams(model);
      % Fit k'th class conditional to k'th random portion of the data.
      % Fit mixing weights to be random.
      % (This is like initializing the assignments randomyl and then
      % doing an M step)
      n = size(data,1);
      model.mixingDistrib = mkRndParams(model.mixingDistrib);
      perm = randperm(n);
      batchSize = max(1,floor(n/K));
      for k=1:K
        start = (k-1)*batchSize+1;
        initdata = data(perm(start:start+batchSize-1),:);
        model.distributions{k} = fit(model.distributions{k},'-data',initdata);
        
         
        
        
        
      end
      model = initPrior(model, data);
      %eng.model = model;
       for k=1:K
        [R, p] = chol(model.distributions{k}.Sigma);
       deter = det(model.distributions{k}.Sigma);
       singular =  ~isfinite(deter) || ~isfinite(p) || deter <=eps || ~(p==0);
       if singular, error('init error');end
       end
      
      
      
    end
    
  end % methods

end

