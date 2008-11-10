classdef MvnMcmcInfer < McmcInfer
  % Gibbs / MH sampling in the multivariate normal 
  
  properties
    mu; Sigma;
    SigmaProposal;
  end
 
  methods
    function eng = MvnMcmcInfer(varargin)
      [mu, Sigma, Nsamples, Nburnin, thin, method, SigmaProposal] = process_options(varargin, ...
        'mu', [], 'Sigma', [], 'Nsamples', 1000, 'Nburnin', 100, 'thin', 1, ...
        'method', [], 'SigmaProposal', []);
      eng.mu = mu; eng.Sigma = Sigma; 
      eng.Nsamples = Nsamples; eng.Nburnin = Nburnin; eng.thin = thin;
      eng.method = method;
      eng.SigmaProposal = SigmaProposal;
      eng.seeds = 1; % 2007b bug - does not call superclass constructor to initialize this
    end
    
    function d = ndims(eng)
      d = length(eng.mu);
    end
    
    function eng = setParams(eng, params)
      eng.mu = params{1};
      eng.Sigma = params{2};
    end
    %{
     function eng = setParams(eng, mu, Sigma)
      eng.mu = mu; eng.Sigma = Sigma;
    end
    %}
    
     function X = gibbsSample2(eng) % for debugging
       % X(i,:) = sample for i=1:n
       n = eng.Nsamples;
       if ~eng.evidenceEntered, error('must call enterEvidence first'); end
       fprintf('%d samples burnin %d\n', n, eng.Nburnin)
       mu = eng.mu; Sigma = eng.Sigma; d = length(mu);
       X = zeros(n, d);
       x = mvnrnd(mu, Sigma); % initial state of chain
       keep = 1;
       V = eng.visVars; H = setdiff(1:d, V);
       x(V) = eng.visValues;
       for iter=1:(n+eng.Nburnin)
         for i=H(:)' % only resample hidden nodes
           visVars = 1:d; visVars(i)=[]; % faster than setdiff(1:d, i)
           [muAgivenB, SigmaAgivenB] = gaussianConditioning(mu, Sigma, visVars, x(visVars));
           x(i) = normrnd(muAgivenB, sqrt(SigmaAgivenB));
         end
         if iter > eng.Nburnin
           X(keep,:) = x; keep = keep + 1;
         end
       end
     end
     
     function x = initChain(eng)
       mu = eng.mu; Sigma = eng.Sigma; d = length(mu);
       %x = mvnrnd(mu, Sigma); 
       x = mu + randn(d,1).*sqrt(diag(Sigma)); 
       V = eng.visVars; H = setdiff(1:d, V);
       x(V) = eng.visValues;
     end
    
     %{
    function CPD = makeFullConditionals(eng)
      mu = eng.mu; Sigma = eng.Sigma; d = length(mu);
      V = eng.visVars; H = setdiff(1:d, V);
      for i=1:d
        if ismember(i,H) % only sample hidden nodes
          CPD{i} = @(x) sampleFullCond(eng, x,i);
        end
      end
    end
    %}
     
    
    function xi = sampleFullCond(eng, x,i)
      mu = eng.mu; Sigma = eng.Sigma; d = length(mu);
      % everything is fully visible except i
      visVars = 1:d; visVars(i)=[]; % faster than setdiff(1:d, i)
      [muAgivenB, SigmaAgivenB] = gaussianConditioning(mu, Sigma, visVars, x(visVars));
      xi = normrnd(muAgivenB, sqrt(SigmaAgivenB));
    end

    function [xprime, probOldToNew, probNewToOld] = proposal(eng, x)
      probOldToNew = 1; probNewToOld = 1; % symmetric proposal
      H = eng.hidVars; V = eng.visVars;
      xprime = x;
      xprime(H) = mvnrnd(x(H), eng.SigmaProposal(H,H));
    end
     
    function  logp = target(eng, x)
      logp = log(mvnpdf(x(:)', eng.mu(:)', eng.Sigma));
    end
    
  end % methods

  
end