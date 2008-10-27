classdef mvnDist < vecDist 
  % multivariate normal p(X|mu,Sigma) 
  
  properties
    mu;
    Sigma;
  end
  
  %% main methods
  methods
    function m = mvnDist(mu, Sigma)
      % mvnDist(mu, Sigma)
      % mu can be a matrix or a pdf, eg. 
      % mvnDist(mvnInvWishDist(...), [])
      if nargin == 0
        mu = []; Sigma = [];
      end
      m.mu  = mu;
      m.Sigma = Sigma;
      m.stateInfEng = mvnExactInfer;
    end

    function params = getModelParams(obj)
      params = {obj.mu, obj.Sigma};
    end
    
    function objS = convertToScalarDist(obj)
      if ndims(obj) ~= 1, error('cannot convert to scalarDst'); end
      objS = gaussDist(obj.mu, obj.Sigma);
    end
    
    function obj = mkRndParams(obj, d)
      if nargin < 2, d = ndims(obj); end
      obj.mu = randn(d,1);
      obj.Sigma = randpd(d);
    end
    
    function d = ndims(m)
      if isa(m.mu, 'double')
        d = length(m.mu);
      else
        d = ndims(m.mu);
      end
    end

    function logZ = lognormconst(obj)
      d = ndims(obj);
      logZ = (d/2)*log(2*pi) + 0.5*logdet(obj.Sigma);
    end
    
    function L = logprob(obj, X)
      % L(i) = log p(X(i,:) | params)
      mu = obj.mu(:)'; % ensure row vector
      if length(mu)==1
        X = X(:); % ensure column vector
      end
      [N d] = size(X);
      if length(mu) ~= d
        error('X should be N x d')
      end
      %if statsToolboxInstalled
      %  L1 = log(mvnpdf(X, obj.mu, obj.Sigma));
      M = repmat(mu, N, 1); % replicate the mean across rows
      if obj.Sigma==0
        L = repmat(NaN,N,1);
      else
        mahal = sum(((X-M)*inv(obj.Sigma)).*(X-M),2);
        L = -0.5*mahal - lognormconst(obj);
      end
      %assert(approxeq(L,L1))
    end
    
    %{
    function h=plotContour2d(obj, varargin)
      % Plot an ellipse representing the 95% contour of a Gaussian
      % eg figure; plotContour2d(mvnDist([0 0], [2 1; 1 1]))
      checkParamsAreConst(obj)
      if ndims(obj) ~= 2
        error('only works for 2d')
      end
      h = gaussPlot2d(obj.mu, obj.Sigma);
    end
     %}
  
    function mu = mean(m)
      checkParamsAreConst(m)
      mu = m.mu;
    end

    function mu = mode(m)
      mu = mean(m);
    end

    function C = cov(m)
      checkParamsAreConst(m)
      C = m.Sigma;
    end
  
    function v = var(obj)
      v = cov(obj);
    end
    
    
%     function samples = sample(obj,n)
%     % Sample n times from this distribution: samples is of size
%     % nsamples-by-ndimensions
%        if(nargin < 2), n = 1; end;
%        A = chol(obj.Sigma,'lower');
%        Z = randn(length(obj.mu),n);
%        samples = bsxfun(@plus,obj.mu(:), A*Z)';
%     end
    
    function obj = fit(obj, varargin)
      % Point estimate of parameters
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % data - data(i,:) = case i
      % suffStat -
      % method - one of {map, mle, covshrink}
      %
      % For covshrink: we use the Ledoit-Wolf formula to estimate srhinkage amount
      %  See  J. Schaefer and K. Strimmer.  2005.  A shrinkage approach to
      %   large-scale covariance matrix estimation and implications
      %   for functional genomics. Statist. Appl. Genet. Mol. Biol. 4:32.

      [X, SS, method] = process_options(...
        varargin, 'data', [], 'suffStat', [], 'method', 'mle');
      hasMissingData =  any(isnan(X(:)));
      assert(~hasMissingData)
      if isempty(SS), SS = mvnDist.mkSuffStat(X); end
      switch method
        case 'mle'
          obj.mu = SS.xbar;
          obj.Sigma = SS.XX;
        case 'covshrink',
          obj.mu =  mean(X);
          obj.Sigma =  covshrinkKPM(X);
        otherwise
          error(['bad method ' method])
      end
    end

    function obj = inferParams(obj, varargin)
      % Computer posterior over params
       % m = inferParams(model, 'name1', val1, 'name2', val2, ...)
       % Arguments are
       % data - data(i,:) = case i
       % suffStat - 
       %
      % if m.mu is of type mvnInvWishDist, and there is no missing
      % data, we compute the posterior exactly. Otherwise we call
      % m = infer(m.paramInfEng, m, data) to deal with it.
      %
       [X, SS] = process_options(...
         varargin, 'data', [], 'suffStat', []);
       hasMissingData =  any(isnan(X(:)));
       if hasMissingData
         obj = infer(obj.paramInfEng, obj, X);
         return;
       end
       if isempty(SS), SS = mvnDist.mkSuffStat(X); end
       if SS.n == 0, return; end
       done = false;
       switch class(obj.mu)
         case 'mvnDist'
           if isa(obj.Sigma, 'double')
             obj.mu = updateMean(obj.mu, SS, obj.Sigma);
             done = true;
           end
         case 'mvnInvWishartDist'
           obj.mu = updateMeanCov(obj.mu, SS);
           done = true;
         case 'double'
           if isa(obj.Sigma, 'invWishartDist')
             obj.Sigma = updateSigma(obj.Sigma, obj.mu, SS);
             done = true;
           end
       end
       if ~done
         obj = infer(obj.paramInfEng, obj, X);
       end
    end
    
    function [postmu, logevidence] = softCondition(pmu, py, A, y)
      % Bayes rule for MVNs
      Syinv = inv(py.Sigma);
      Smuinv = inv(pmu.Sigma);
      postSigma = inv(Smuinv + A'*Syinv*A);
      postmu = postSigma*(A'*Syinv*(y-py.mu) + Smuinv*pmu.mu);
      postmu = mvnDist(postmu, postSigma);
      %evidence = mvnpdf(y(:)', (A*pmu.mu + py.mu)', py.Sigma + A*pmu.Sigma*A');
      logevidence = logprob(mvnDist(A*pmu.mu + py.mu, py.Sigma + A*pmu.Sigma*A'), y(:)');
    end
    
  end % methods

  %% Demos
  methods(Static = true)
    function suffStat = mkSuffStat(X)
      % SS.n
      % SS.xbar = 1/n sum_i X(i,:)'
      % SS.XX(j,k) = 1/n sum_i XC(i,j) XC(i,k)
      n = size(X,1);
      suffStat.n = n;
      %suffStat.X = sum(X,1)'; % column vector
      suffStat.xbar = sum(X,1)'/n; % column vector
      Xc = (X-repmat(suffStat.xbar',n,1));
      suffStat.XX = (Xc'*Xc)/n;
    end

     function demoSoftCondition()
      setSeed(0);
      muTrue = [0.5 0.5]'; Ctrue = 0.1*[2 1; 1 1];
      mtrue = mvnDist(muTrue, Ctrue);
      n = 10;
      X = sample(mtrue, n);
      prior = mvnDist([0 0]', 0.1*eye(2));
      A = repmat(eye(2), n, 1);
      py = mvnDist(zeros(2*n,1), kron(eye(n), Ctrue));
      data = X'; y = data(:);
      postMu = softCondition(prior, py, A, y); 
      m = mvnDist(prior, Ctrue);
      m = inferParams(m, 'data', X(1:n,:));
      assert(approxeq(postMu.mu, m.mu.mu))
      assert(approxeq(postMu.Sigma, m.mu.Sigma))
     end


    function demoSeqUpdateMuSigma1d(nu, S)
     if nargin < 1, nu = 0.001; S = 0.001; end
      setSeed(1);
      muTrue = 5; varTrue = 10;
      X = sample(mvnDist(muTrue, varTrue), 500);
      muRange = [-5 15]; sigmaRange  = [0.1 15];
      figure; hold on;
      [styles, colors, symbols] =  plotColors();
      ns = [0 2 5 50]
      for i=1:length(ns)
        k = 0.001;
        prior = mvnInvWishartDist('mu', 0, 'k', k, 'dof', nu, 'Sigma', S);
        n = ns(i);
        m = inferParams(mvnDist(prior, []), 'data', X(1:n));
        post = m.mu;
        [h(i), ps{i}] = plot(post, 'plotArgs', {styles{i}, 'linewidth', 2}, ...
          'xrange', [muRange sigmaRange], 'useContour', true);
        legendstr{i} = sprintf('n=%d', n); 
        xbar = mean(X(1:n)); vbar = var(X(1:n));
        h(i)=plot(xbar, vbar, 'x','color',colors(i),'markersize', 12,'linewidth',3);
      end
      xlabel(sprintf('%s', '\mu'))
      ylabel(sprintf('%s', '\sigma^2'))
      legend(h,legendstr);
      title(sprintf('prior = NIW(mu=0, k=%5.3f, %s=%5.3f, S=%5.3f), true %s=%5.3f, %s=%5.3f', ...
        k, '\nu', nu, S, '\mu', muTrue, '\sigma^2', varTrue))
    end

    
    function [prior, post] = demoInferParamsMuSigma1dPriors()
      seed = 0; rand('state', seed); randn('state', seed);
      muTrue = 10; varTrue = 5^2;
      N = 12;
      X = sample(mvnDist(muTrue, varTrue), N);
      %X = normrnd(muTrue, sqrt(varTrue), N, 1);
      %X = [141, 102, 73, 171, 137, 91, 81, 157, 146, 69, 121, 134];
      v = 1; S = var(X);
      prior{1} = mvnInvWishartDist('mu', mean(X), 'k', 1, 'dof', v, 'Sigma', v*S);
      names{1} = 'Data-driven'; % since has access to data
      v = 0; S = 0;
      prior{2} = mvnInvWishartDist('mu', 0, 'k', 0.01, 'dof', v, 'Sigma', v*S);
      names{2} = 'Jeffreys'; % Jeffrey
      v = N; S = 10;
      prior{3} = mvnInvWishartDist('mu', 5, 'k', N, 'dof', v, 'Sigma', v*S);
      names{3} = 'Wrong';
      muRange = [0 20]; sigmaRange  = [1 30];
      nr = 3; nc = 3;
      figure;
      for i=1:3
        m = inferParams(mvnDist(prior{i}, []), 'data', X);
        post{i} = m.mu;
        pmuPost = marginal(post{i}, 'mu');
        pSigmaPost = marginal(post{i}, 'Sigma');
        pmuPrior = marginal(prior{i}, 'mu');
        pSigmaPrior = marginal(prior{i}, 'Sigma');

        subplot2(nr,nc,i,1);
        plot(pmuPrior, 'plotArgs', {'k:', 'linewidth',2}, 'xrange', muRange); hold on
        plot(pmuPost, 'plotArgs', {'r-', 'linewidth', 2}, 'xrange', muRange);
        title(sprintf('p(%s|D) %s', '\mu', names{i}(1)))
        %legend('prior', 'post')

        subplot2(nr,nc,i,2);
        plot(pSigmaPrior, 'plotArgs', {'k:', 'linewidth',2}, 'xrange', sigmaRange); hold on
        plot(pSigmaPost, 'plotArgs', {'r-', 'linewidth', 2}, 'xrange', sigmaRange);
        title(sprintf('p(%s|D) %s', '\sigma^2', names{i}(1)))

        subplot2(nr,nc,i,3);
        plot(prior{i}, 'plotArgs', {'k:', 'linewidth',2}, ...
          'xrange', [muRange sigmaRange], 'useContour', true); hold on
        plot(post{i}, 'plotArgs', {'r-', 'linewidth', 2}, ...
          'xrange', [muRange sigmaRange], 'useContour', true);
        %title(sprintf('p(%s,%s|D) %s', '\mu', '\sigma^2', names{i}));
        title(sprintf('%s', names{i}));
        %xlabel(sprintf('%s','\mu')); ylabel(sprintf('%s','\sigma^2'));
      end
    end

     function demoSeqUpdateSigma1d(nu, S)
       if nargin < 1, nu = 0.001; S = 0.001; end
      setSeed(1);
      mutrue = 5; Ctrue = 10;
      mtrue = mvnDist(mutrue, Ctrue);
      n = 500;
      X = sample(mtrue, n);
      ns = [0 2 5 50]
      fig1= figure; hold on;
      fig2 = figure; 
      pmax = -inf;
      [styles, colors, symbols] =  plotColors();
      for i=1:length(ns)
        prior = invWishartDist(nu, S);
        n = ns(i);
        m = inferParams(mvnDist(mutrue, prior), 'data', X(1:n));
        post = m.Sigma;
        mean(post);
        figure(fig1);
        [h(i), p]= plot(post, 'plotArgs', {styles{i}, 'linewidth', 2}, 'xrange', [0 15]);
        legendstr{i} = sprintf('n=%d', n);
        pmax = max(pmax, max(p));
        xbar = mean(X(1:n)); vbar = var(X(1:n));
        %hh(i)=line([vbar vbar], [0 pmax],'color',colors(i),'linewidth',3);
        
        if nu<1 && n==0, continue; end % improper prior, cannot sample from it
        figure(fig2); subplot(length(ns),1,i);
        XX = sample(post,100);
        hist(XX)
        title(legendstr{i})
      end
      figure(fig1);
      legend(h,legendstr);
      titlestr = sprintf('prior = IW(%s=%5.3f, S=%5.3f), true %s=%5.3f', ...
        '\nu', nu, S, '\sigma^2', Ctrue);
      title(titlestr)
      line([Ctrue Ctrue], [0 pmax],'color','k','linewidth',3);
      figure(fig2); suptitle(titlestr);
     end
     
      function demoSeqUpdateMu1d()
      setSeed(1);
      mutrue = 5; Ctrue = 10;
      mtrue = mvnDist(mutrue, Ctrue);
      n = 500;
      X = sample(mtrue, n);
      ns = [0 2 5 50]
      figure; hold on;
      pmax = -inf;
      [styles, colors, symbols] =  plotColors();
      for i=1:length(ns)
        k = 0.001;
        prior = mvnDist(0, 1/k);
        n = ns(i);
        m = inferParams(mvnDist(prior, Ctrue), 'data', X(1:n));
        post = m.mu;
        [h(i), p]= plot(post, 'plotArgs', {styles{i}, 'linewidth', 2}, 'xrange', [0 10]);
        legendstr{i} = sprintf('n=%d', n);
        pmax = max(pmax, max(p));
        xbar = mean(X(1:n)); vbar = var(X(1:n));
        %h(i)=line([xbar xbar], [0 pmax],'color',colors(i),'linewidth',3);
      end
      legend(h,legendstr);
      title(sprintf('prior = N(mu0=0, v0=%5.3f), true %s = %5.3f', 1/k, '\mu', mutrue))
      line([mutrue, mutrue], [0 pmax],'color','k','linewidth',3);
     end
     
    function demoInferParamsSigma2d(doSave)
      if nargin < 1, doSave = false; end
      folder = 'C:\kmurphy\PML\pdfFigures';
      seed = 0; randn('state', seed); rand('state', seed);
      muTrue = [0 0]'; Ctrue = 0.1*[2 1; 1 1];
      mtrue = mvnDist(muTrue, Ctrue);
      xrange = 2*[-1 1 -1 1];
      n = 20;
      X = sample(mtrue, n);
      ns = [20];
      figure;
      useContour = true;
      plot(X(:,1), X(:,2), '.', 'markersize',15);
      axis(xrange);
      hold on
      %plotContour2d(mtrue);
      gaussPlot2d(mtrue.mu, mtrue.Sigma);
      title('truth'); grid on;
      fname = fullfile(folder, sprintf('MVNcovDemoData.pdf'));
      if doSave, pdfcrop; print(gcf, '-dpdf', fname); end

      %prior = invWishartDist(10, Ctrue); % cheat!
      prior = invWishartDist(2, eye(2));
      plotMarginals(prior);
      %set(gcf, 'name', 'prior');
      suplabel('prior');
      fname = fullfile(folder, sprintf('MVNcovDemoPriorMarg.pdf'));
      if doSave, pdfcrop; print(gcf, '-dpdf', fname); end

      plotSamples2d(prior, 9);
      subplot(3,3,1); gaussPlot2d(mtrue.mu, mtrue.Sigma);  title('truth');
      suplabel('prior');
      fname = fullfile(folder, sprintf('MVNcovDemoPriorSamples.pdf'));
      if doSave, pdfcrop; print(gcf, '-dpdf', fname); end

      for i=1:length(ns)
        n = ns(i);
        m = mvnDist(muTrue, prior);
        m = inferParams(m, 'data', X(1:n,:));
        post = m.Sigma;
        plotMarginals(post);
        suplabel(sprintf('post after %d obs', n));
        fname = fullfile(folder, sprintf('MVNcovDemoPost%dMarg.pdf', n));
        if doSave, pdfcrop; print(gcf, '-dpdf', fname); end

        plotSamples2d(post, 9);
        subplot(3,3,1); gaussPlot2d(mtrue.mu, mtrue.Sigma); title('truth');
        suplabel(sprintf('post after %d obs', n));
        fname = fullfile(folder, sprintf('MVNcovDemoPost%dSamples.pdf', n));
        if doSave, pdfcrop; print(gcf, '-dpdf', fname); end
      end
    end

    function demoInferParamsMean2d()
      setSeed(0);
      muTrue = [0.5 0.5]'; Ctrue = 0.1*[2 1; 1 1];
      mtrue = mvnDist(muTrue, Ctrue);
      xrange = [-1 1 -1 1];
      n = 10;
      X = sample(mtrue, n);
      ns = [2 5 10];
      figure;
      useContour = true;
      nr = 2; nc = 3;
      subplot(nr,nc,1);
      plot(X(:,1), X(:,2), '.', 'markersize',15);
      %hold on; for i=1:n, text(X(i,1), X(i,2), sprintf('%d', i)); end
      axis(xrange); title('data'); grid on; axis square
      subplot(nr,nc,2);
      plot(mtrue, 'xrange', [-1 2 -1 2], 'useContour', useContour);
      %gaussPlot2d(mtrue.mu, mtrue.Sigma);
      title('truth'); grid on; axis square
      prior = mvnDist([0 0]', 0.1*eye(2));
      subplot(nr,nc,3); plot(prior, 'xrange', xrange, 'useContour', useContour);
      title('prior'); grid on; axis square
      for i=1:length(ns)
        n = ns(i);
        m = mvnDist(prior, Ctrue);
        m = inferParams(m, 'data', X(1:n,:));
        post = m.mu;
        subplot(nr,nc,i+3); plot(post, 'xrange', xrange, 'useContour', useContour, 'npoints', 150);
        title(sprintf('post after %d obs', n)); grid on; axis square
      end
    end

   
    function demoInferParamsMean1d()
      mvnDist.helperInferParamsMean1d(1);
      mvnDist.helperInferParamsMean1d(5);
    end
      
    function helperInferParamsMean1d(priorVar)
      if nargin < 1, priorVar = 1; end
      prior = mvnDist(0, priorVar);
      sigma2 = 1;
      m = mvnDist(prior, sigma2);
      x = 3;
      m = inferParams(m, 'data', x);
      post = m.mu;
      % The likelihood is proportional to the posterior when we use a flat prior
      priorBroad = mvnDist(0, 1e10);
      m2 = mvnDist(priorBroad, sigma2);
      m2 = inferParams(m2, 'data', x);
      lik = m2.mu;
      % Now plot
      figure;
      xrange = [-5 5];
      hold on
      plot(prior, 'xrange', xrange, 'plotArgs', { 'r-', 'linewidth', 2});
      legendstr{1} = 'prior';
      plot(lik, 'xrange', xrange,'plotArgs', {'k:o', 'linewidth', 2});
      legendstr{2} = 'lik';
      plot(post, 'xrange', xrange,'plotArgs', {'b-.', 'linewidth', 2});
      legendstr{3} = 'post';
      legend(legendstr)
      title(sprintf('prior variance = %3.2f', priorVar))
    end

    function demoCondition()
      setSeed(0);
      d = 4;
      obj = mkRndParams(mvnDist, d);
      x = randn(d,1);
      V = [3 4]; 
      obj = enterEvidence(obj, V, x(V));
      x(:)'
      fprintf('j \t %8s \t %8s\n', 'mean', 'Var');
      for j=1:d
        m = marginal(obj, j);
        fprintf('%d \t %8.3f \t %8.3f\n', j, mean(m), var(m));
      end
    end
    
    function demoCondition2d()
      % Take a horizontal slice thru a 2d Gaussian and plot the resulting
      % conditional
      mu = [0 0]';
      rho = 0.5;
      %S  = [4 1; 1 1];
      S = [1 rho; rho 1];
      obj = mvnDist(mu,S);
      figure;
      gaussPlot2d(obj.mu, obj.Sigma);
      hold on;
      [U,D] = eig(S); %plot
      sf=  -2.5;
      line([mu(1) mu(1)+sf*sqrt(D(1,1))*U(1,1)],[mu(2) mu(2)+sf*sqrt(D(1,1))*U(2,1)],'linewidth',2)
      line([mu(1) mu(1)+sf*sqrt(D(2,2))*U(1,2)],[mu(2) mu(2)+sf*sqrt(D(2,2))*U(2,2)],'linewidth',2)
      %line([-5 5], [-5 5]);
      x2 = 1; line([-5 5], [x2 x2],  'color', 'r', 'linewidth', 2);
      %post = conditional(obj, 2, x2); % 2 is the y axis
      obj = enterEvidence(obj, 2, x2);
      post = marginal(obj, 1);
      %plot(post, 'xrange', [-4 4]);
      xs = -5:0.1:5;
      ps = exp(logprob(post, xs));
      ps = 50*normalize(ps);
      plot(xs, 1+ps, 'k', 'linewidth',2 );
      postMu = mean(post);
      line([postMu postMu], [-4 4], 'color', 'k', 'linewidth', 2, 'linestyle', '-');
      %grid on
      title(sprintf('p(x1,x2)=N([0 0], [1 %3.2f; %3.2f 1]), p(x1|x2=%3.1f)=N(x1|%3.2f, %3.2f)', ...
        rho, rho, x2, mean(post), var(post)));
    end

    function demoPlot2dMarginals()
      plotGauss2dMargCond;
    end

    function demoPlot2d(doSave)
      if nargin < 1, doSave = false; end
      mu = [1 0]';  S  = [2 1.8; 1.8 2]
      folder = 'C:\kmurphy\PML\pdfFigures';
      figure; plot(mvnDist(mu,S), 'xrange', [-6 6 -6 6]); title('full');
      if doSave, pdfcrop; print(gcf, '-dpdf', fullfile(folder, 'gaussPlot2dDemoSurfFull')); end
      figure; plot(mvnDist(mu,S), 'xrange', [-6 6 -6 6], 'useContour', true); title('full');
      if doSave, pdfcrop; print(gcf, '-dpdf', fullfile(folder, 'gaussPlot2dDemoContourFull')); end
      [U,D] = eig(S);
      % Decorrelate
      S1 = U'*S*U
      figure; plot(mvnDist(mu,S1), 'xrange', [-5 5 -10 10]); title('diagonal')
      if doSave, pdfcrop; print(gcf, '-dpdf', fullfile(folder, 'gaussPlot2dDemoSurfDiag')); end
      figure; plot(mvnDist(mu,S1), 'xrange', [-5 5 -10 10], 'useContour', true); title('diagonal');
      if doSave, pdfcrop; print(gcf, '-dpdf', fullfile(folder, 'gaussPlot2dDemoContourDiag')); end
      % Compute whitening transform:
      A = sqrt(inv(D))*U';
      mu2 = A*mu;
      S2  = A*S*A' % might not be numerically equal to I
      assert(approxeq(S2, eye(2)))
      S2 = eye(2); % to ensure picture is pretty
      % we plot centered on original mu, not shifted mu
      figure; plot(mvnDist(mu,S2), 'xrange', [-5 5 -5 5]); title('spherical');
      if doSave, pdfcrop; print(gcf, '-dpdf', fullfile(folder, 'gaussPlot2dDemoSurfSpherical')); end
      figure; plot(mvnDist(mu,S2), 'xrange', [-5 5 -5 5], 'useContour', true);
      title('spherical');axis('equal');
      if doSave, pdfcrop; print(gcf, '-dpdf', fullfile(folder, 'gaussPlot2dDemoContourSpherical')); end
    end

    function demoHeightWeight()
      mvnDist.heightWeightHelper(false);
      mvnDist.heightWeightHelper(true);
    end

    function heightWeightHelper(plotCov)
      if nargin < 1, plotCov = true; end
      % Make 2D scatter plot and superimpose Gaussian fit
      rawdata = dlmread('heightWeightDataSimple.txt'); % comma delimited file
      data.Y = rawdata(:,1); % 1=male, 2=female
      data.X = [rawdata(:,2) rawdata(:,3)]; % height, weight
      maleNdx = find(data.Y == 1);
      femaleNdx = find(data.Y == 2);
      classNdx = {maleNdx, femaleNdx};
      figure;
      colors = 'br';
      sym = 'xo';
      for c=1:2
        str = sprintf('%s%s', sym(c), colors(c));
        X = data.X(classNdx{c},:);
        h=scatter(X(:,1), X(:,2), 100, str); %set(h, 'markersize', 10);
        hold on
        if plotCov
          pgauss = fit(mvnDist, 'data',X, 'method', 'mle');
          gaussPlot2d(pgauss.mu, pgauss.Sigma);
        end
      end
      xlabel('height')
      ylabel('weight')
      title('red = female, blue=male')
    end

    function demoImputation(varargin)
      demoImputation@vecDist(mvnDist, varargin{:});
    end

  end


  %% Private methods
  methods(Access = 'protected')
    function checkParamsAreConst(obj)
      p = isa(obj.mu, 'double') && isa(obj.Sigma, 'double');
      if ~p
        error('params must be constant')
      end
    end
  end


end