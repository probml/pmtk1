classdef mvnMcmcInfer < mcmcInfer
  % Gibbs / MH sampling in the multivariate normal 
  
  properties
    mu; Sigma;
    SigmaProposal;
  end
 
  methods
    function eng = mvnMcmcInfer(varargin)
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

   %% Static
  methods(Static = true)
     
    function demo(seed)
      if nargin < 1, seed = 1; end
      setSeed(seed);
      m = mvnDist;
      d = 5;
      m = mkRndParams(m, d);
      x = randn(1,d);
      V = [1 2]; H = setdiff(1:d, V);
      model{1} = m; model{1}.stateInfEng = mvnExactInfer;
      name{1}= 'exact';
      model{2} = m; model{2}.stateInfEng = mvnMcmcInfer('method', 'gibbs');
      name{2} = 'gibbs';
      %model{3} = m; model{3}.stateInfEng = mvnMcmcInfer('method', 'gibbs2');
      %name{3} = 'gibbs2';
      model{3} = m; model{3}.stateInfEng = mvnMcmcInfer('method', 'mh', 'SigmaProposal', 0.1*m.Sigma);
      name{3} = 'mh';
      Nmethods = 3;
      for i=1:Nmethods
        model{i} = enterEvidence(model{i}, V, x(V));
        query{i} = marginal(model{i}, H);
        fprintf('method %s\n', name{i})
        mu = mean(query{i})
        C = cov(query{i})
      end
    end
    
    
   

    function demo2d()
      mvnMcmcInfer.helper2d('exact');
      mvnMcmcInfer.helper2d('gibbs');
      mvnMcmcInfer.helper2d('mhI');
      mvnMcmcInfer.helper2d('mhI01');
    end

    function helper2d(method)
      % sample from a 2d Gaussian and compare to exact distribution
      m = mvnDist;
      setSeed(0);
      m.Sigma = [1 -0.5; -0.5 1];
      m.mu = [1; 1];
      %m = enterEvidence(m, [], []);
      for i=1:2
        margExact{i} = marginal(m,i);
      end
      N = 500;
      switch lower(method)
        case 'exact'
          m.stateInfEng = mvnExactInfer;
        case 'gibbs'
          m.stateInfEng = mvnMcmcInfer('method', 'gibbs', 'Nsamples', N);
        case 'mhi'
          m.stateInfEng = mvnMcmcInfer('method', 'mh', 'SigmaProposal', eye(2), ...
            'Nsamples', N, 'Nburnin', 500);
        case 'mhi01'
          m.stateInfEng = mvnMcmcInfer('method', 'mh', 'SigmaProposal', 0.01*eye(2), ...
            'Nsamples', N, 'Nburnin', 500);
        case 'mhtrue'
          m.stateInfEng = mvnMcmcInfer('method', 'mh', 'SigmaProposal', m.Sigma, ...
            'Nsamples', N, 'Nburnin', 500);
        otherwise
          error(['unknown method ' method])
      end
      %m = enterEvidence(m, [], []);
      X = sample(m, N);
      mS = sampleDist(X); % convert samples to a distribution
      for i=1:2
        margApprox{i} = marginal(mS,i);
      end
      figure;
      plot(m, 'useContour', 'true');
      hold on
      plot(mS);
      title(sprintf('%s', method))
     
      figure;
      plot(m, 'useContour', 'true');
      hold on
      plot(X(:,1), X(:,2), 'o-');
      title(method)
      
      figure;
      for i=1:2
        subplot2(2,2,i,1);
        [h, histArea] = plot(margApprox{i}, 'useHisto', true);
        hold on
        [h, p] = plot(margExact{i}, 'scaleFactor', histArea, ...
          'plotArgs', {'linewidth', 2, 'color', 'r'});
        title(sprintf('exact mean=%5.3f, var=%5.3f', mean(margExact{i}), var(margExact{i})));
        subplot2(2,2,i,2);
        plot(margApprox{i}, 'useHisto', false);
        title(sprintf('approx mean=%5.3f, var=%5.3f', mean(margApprox{i}), var(margApprox{i})));
      end
      
      % convergence diagnostics
      if ~strcmpi(method, 'exact')
        m.stateInfEng.seeds = 1:3;
        m.stateInfEng.Nburnin = 0;
        N = 1000;
        m.stateInfEng.Nsamples = N;
        % we must call enterEvidence since the params (Nsamples, seeds)
        % have changed; otherwise we will use the old values
        m = enterEvidence(m,[],[]);
        X  = sample(m, N); % X(sample, dim, chain)
        mcmcInfer.plotConvDiagnostics(X, 1, sprintf('%s', method));
      end
    end

   
  end
  
end