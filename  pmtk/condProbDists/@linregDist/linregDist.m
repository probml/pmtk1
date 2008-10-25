classdef linregDist < condProbDist
  
  properties
    w; 
    sigma2;
    transformer;
  end
  
  %% Main methods
  methods 
    function m = linregDist(varargin)
      [transformer, w, sigma2] = process_options(...
        varargin, 'transformer', [],  'w', [], 'sigma2', []);
      m.transformer = transformer;
      m.w = w;
      m.sigma2 = sigma2;
    end
     
     function obj = mkRndParams(obj, d)
       obj.w = randn(d,1);
       obj.sigma2 = rand(1,1);
     end
    
     function p = logprob(obj, X, y)
       % p(i) = log p(Y(i,:) | X(i,:), params)
       [yhat] = mean(predict(obj, X));
       s2 = obj.sigma2;
       p = -1/(2*s2)*(y(:)-yhat(:)).^2 - 0.5*log(2*pi*s2);
       %[yhat, py] = predict(obj, X);
       %PP = logprob(py, y); % PP(i,j) = p(Y(i)| yhat(j))
       %p1 = diag(PP);
       %yhat = predict(obj, X);
       %assert(approxeq(p,p1))
     end

      function p = squaredErr(obj, X, y)
        yhat = mean(predict(obj, X));
        p  = (y(:)-yhat(:)).^2;
      end
     
      function py = predict(obj, X)
       if ~isempty(obj.transformer)
         X = test(obj.transformer, X);
       end
       n = size(X,1);
       muHat = X*obj.w;
       sigma2Hat = obj.sigma2*ones(n,1); % constant variance!
       py = gaussDist(muHat, sigma2Hat);
      end
      
      function [py] = postPredict(obj, X)
        if ~isempty(obj.transformer)
          X = test(obj.transformer, X);
        end
        n = size(X,1);
        done = false;
         switch class(obj.w)
          case 'mvnDist'
            if isa(obj.sigma2, 'double')
              muHat = X*obj.w.mu;
              Sn = obj.w.Sigma;
              sigma2Hat = obj.sigma2*ones(n,1) + diag(X*Sn*X');
              %{
              for i=1:n
                xi = X(i,:)';
                s2(i) = obj.sigma2 + xi'*Sn*xi;
              end
              assert(approxeq(sigma2Hat, s2))
              %}
              py = gaussDist(muHat, sigma2Hat);
              done = true;
            end
           case 'mvnInvGammaDist'
             wn = obj.w.mu; 
             Sn = obj.w.Sigma;
             vn = obj.w.a*2;
             sn2 = 2*obj.w.b/vn;
             m = size(X,n);
             SS = sn2*(eye(m) + X*Sn*X');
             py = studentDist(vn, X*wn, diag(SS));
             done = true;
         end
         assert(done)
      end
     
      function obj = inferParams(obj, varargin)
        % m = inferParams(model, 'name1', val1, 'name2', val2, ...)
        % Arguments are
        % 'X' - X(i,:) Do NOT include a column of 1's
        % 'y'- y(i)
        % lambda >= 0
        % 'prior' - one of {mvnDist object, mvnInvGammaDist object, ...
        %                   'mvn', 'mvnIG'}
        % In the latter 2 cases, we create a diagonal Gaussian prior
        % with precision lambda (except for the offset term)
        [X, y, lambda, prior, sigma2] = process_options(...
          varargin, 'X', [], 'y', [], 'lambda', 1e-3, 'prior', 'mvn',...
          'sigma2', []);
        if ~isempty(obj.transformer)
          [X, obj.transformer] = train(obj.transformer, X);
        end
        if isa(prior, 'char')
          obj.w = makeSphericalPrior(obj, X, lambda, prior);
        end
        if ~isempty(sigma2)
          % this is ignored if the prior is mvnIG
          obj.sigma2 = sigma2; 
        end
        done = false;
        switch class(obj.w)
            case 'mvnDist'
            if isa(obj.sigma2, 'double') && obj.sigma2 > 0
              % conjugate updating of w with fixed sigma2
              S0 = obj.w.Sigma; w0 = obj.w.mu;
              s2 = obj.sigma2; sigma = sqrt(s2);
              Lam0 = inv(S0); % yuck!
              [wn, Sn] = normalEqnsBayes(X, y, Lam0, w0, sigma);
              obj.w = mvnDist(wn, Sn);
              done = true;
            end
          case 'mvnInvGammaDist'
            % conjugate updating with unknown w and sigma2
            obj.w = updateMVNIG(obj, X, y);
           done = true;
        end
        assert(done)
      end
      
     function obj = fit(obj, varargin)
      % m = fit(model, 'name1', val1, 'name2', val2, ...)
      % Arguments are
      % 'X' - X(i,:) Do NOT include a column of 1's
      % 'y'- y(i)
      % 'prior' - one of {'none', 'L2', 'L1'}
      % 'lambda' >= 0
      % method - must be one of { ridgeQR, ridgeSVD }.
      [X, y, method, lambda, prior] = process_options(...
        varargin, 'X', [], 'y', [], 'method', 'ridgeQR', ...
        'lambda', 0, 'prior', 'none');
      if lambda>0 && strcmpi(prior, 'none'), prior = 'L2'; end
      if ~isempty(obj.transformer)
         [X, obj.transformer] = train(obj.transformer, X);
      end
      switch lower(prior)
        case 'none'
          n = size(X,1);
          obj.w = X \ y;
          yhat = X*obj.w;
          obj.sigma2 = mean((yhat-y).^2); % 1/n, not unbiased
          
        case 'l2'
          if ~isempty(obj.transformer) && addOffset(obj.transformer)
            X = X(:,2:end); % remove leading column of 1s
            addOnes = true;
          else
            addOnes = false;
          end
          obj.w = ridgereg(X, y, lambda, method, addOnes);
          n = size(X,1);
          if addOnes
            X = [ones(n,1) X]; % column of 1s for w0 term
          end
          yhat = X*obj.w;
          obj.sigma2 = mean((yhat-y).^2); % 1/n, not unbiased
          
        case 'l1'
          % lasso
          error('not yet implemented')
        otherwise
          error(['unrecognized method ' method])
      end
     end
    
     function s = bicScore(obj, X, y, lambda)
       L = sum(logprob(obj, X, y));
       n = size(X,1);
       %d = length(obj.w);
       d = dofRidge(obj, X, lambda);
       s = L-0.5*d*log(n);
     end
     
     function s = aicScore(obj, X, y, lambda)
       L = sum(logprob(obj, X, y));
       n = size(X,1);
       %d = length(obj.w);
       d = dofRidge(obj, X, lambda);
       s = L-d;
     end
     
     function df = dofRidge(obj, X, lambdas)
       % compute the degrees of freedom for a given lambda value
       % Elements of Statistical Learning p63
       if nargin < 3, lambdas = obj.lambda; end
       if ~isempty(obj.transformer)
         X = train(obj.transformer, X);
         if addOffset(obj.transformer)
           X = X(:,2:end);
         end
       end
       xbar = mean(X);
       XC = X - repmat(xbar,size(X,1),1);
       [U,D,V] = svd(XC,'econ');
       D2 = diag(D.^2);
       for i=1:length(lambdas)
         df(i) = sum(D2./(D2+lambdas(i)));
       end
     end

  end
  
  %% Demos
  methods(Static = true)
    
    function demoPolyFitDegree()
      % based on code code by Romain Thibaux
      % (Lecture 2 from http://www.cs.berkeley.edu/~asimma/294-fall06/)
      %makePolyData;
      %[xtrain, ytrain, xtest, ytest] = makePolyData;
      [xtrain, ytrain, xtest, ytestNoisefree, ytest] = polyDataMake('sampling','thibaux');
      doPrint = 0;
      figure(1);clf
      scatter(xtrain,ytrain,'b','filled');
      %title('true function and noisy observations')
      folder = 'C:\kmurphy\PML\pdfFigures';
      degs = 0:16;
      for i=1:length(degs)
        deg = degs(i);
        m = linregDist;
        m.transformer =  chainTransformer({rescaleTransformer, polyBasisTransformer(deg)});
        m = fit(m, 'X', xtrain, 'y', ytrain);
        ypredTrain = mean(predict(m, xtrain));
        ypredTest = mean(predict(m, xtest));
        testMse(i) = mean((ypredTest - ytest).^2);
        trainMse(i) = mean((ypredTrain - ytrain).^2);
        testLogprob(i) = sum(logprob(m, xtest, ytest));
        trainLogprob(i) = sum(logprob(m, xtrain, ytrain));
        [CVmeanMse(i), CVstdErrMse(i)] = cvScore(m, xtrain, ytrain, ...
          'objective', 'squaredErr');
        [CVmeanLogprob(i), CVstdErrLogprob(i)] = cvScore(m, xtrain, ytrain, ...
          'objective', 'logprob');

        figure(1);clf
        scatter(xtrain,ytrain,'b','filled');
        hold on;
        plot(xtest, ypredTest, 'k', 'linewidth', 3);
        hold off
        title(sprintf('degree %d, train mse %5.3f, test mse %5.3f',...
          deg, trainMse(i), testMse(i)))
        set(gca,'ylim',[-10 15]);
        set(gca,'xlim',[-1 21]);
        if doPrint
          fname = sprintf('%s/polyfitDemo%d.pdf', folder, deg)
          pdfcrop; print(gcf, '-dpdf', fname);
        end
      end


      figure(3);clf
      hold on
      %plot(degs, -CVmeanMse, 'ko-', 'linewidth', 2, 'markersize', 12);
      plot(degs, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
      plot(degs, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
      %errorbar(degs, -CVmeanMse, CVstdErrMse, 'k');
      xlabel('degree')
      ylabel('mse')
      legend('train', 'test')
      if doPrint
        fname = sprintf('%s/polyfitDemoUcurve.pdf', folder)
        pdfcrop; print(gcf, '-dpdf', fname);
      end

    end

    
    function demoPolyFitRidge(n)
      % based on code by Romain Thibaux
      %(Lecture 2 from http://www.cs.berkeley.edu/~asimma/294-fall06/)
      %makePolyData;
      if nargin < 1, n = 21; end
      [xtrain, ytrain, xtest, ytest] = polyDataMake('sampling', 'thibaux','n',n);
      deg = 14;
      m = linregDist();
      m.transformer =  chainTransformer({rescaleTransformer, ...
        polyBasisTransformer(deg)});
      lambdas = [0 0.00001 0.001];
      for k=1:length(lambdas)
        lambda = lambdas(k);
        m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
        format bank
        m.w(:)'
        ypred = mean(predict(m, xtest));
        figure;
        scatter(xtrain,ytrain,'b','filled');
        hold on;
        plot(xtest, ypred, 'k', 'linewidth', 3);
        hold off
        title(sprintf('degree %d, lambda %10.8f', deg, lambda))
      end
    end

    function demoPolyFitRidgeU(n)
      if nargin < 1, n = 21; end
      [xtrain, ytrain, xtest, ytest] = polyDataMake('n', n, 'sampling', 'thibaux');
      deg = 14;
      m = linregDist();
      m.transformer =  chainTransformer({rescaleTransformer,  polyBasisTransformer(deg)});
      lambdas = logspace(-10,1.2,15);
      for k=1:length(lambdas)
        lambda = lambdas(k);
        m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
        testMse(k) = mean(squaredErr(m, xtest, ytest));
        trainMse(k) = mean(squaredErr(m, xtrain, ytrain));
      end
      figure;
      hold on
      ndx = log(lambdas);
      %ndx  = lambdas;
      if 1
        plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
        plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
      else
        semilogx(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
        semilogx(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
      end
      legend('train', 'test')
      %xlabel('dof')
      xlabel('log(lambda)')
      ylabel('mse')
      title(sprintf('poly degree %d, ntrain  %d', deg, n))

      %xx = train(m.transformer, xtrain);
      df = dofRidge(m, xtrain, lambdas);
      figure; hold on;
      %ndx = log(df);
      ndx = df;
      if 1
        plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
        plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
      else
        semilogx(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
        semilogx(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
      end
      legend('train', 'test')
      xlabel('dof')
      ylabel('mse')
      title(sprintf('poly degree %d, ntrain  %d', deg, n))
    end

    function demoPolyFitN(deg)
      linregDist.helperPolyFitN(1);
      linregDist.helperPolyFitN(2);
      linregDist.helperPolyFitN(30);
    end
      
    function helperPolyFitN(deg)
      lambda = 1e-3;
      ns = linspace(10,200,10);
      for i=1:length(ns)
        n=ns(i);
        %[xtrain, ytrain, xtest, ytest] = makePolyData(n);
        [xtrain, ytrain, xtest, ytestNoiseFree, ytest,sigma2] = polyDataMake('n', n, 'sampling', 'thibaux');
        m = linregDist();
        m.transformer =  chainTransformer({rescaleTransformer, polyBasisTransformer(deg)});
        m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
        testMse(i) = mean(squaredErr(m, xtest, ytest));
        trainMse(i) = mean(squaredErr(m, xtrain, ytrain));
      end
      figure;
      hold on
      ndx = ns;
      plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
      plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
      legend('train', 'test')
      ylabel('mse')
      xlabel('size of training set')
      title(sprintf('truth=degree 2, model = degree %d', deg));
      set(gca,'ylim',[0 22]);
      line([0 max(ns)],[sigma2 sigma2],'color','k','linewidth',3);
    end
    
    function demoPolyFitRidgeCV(n)
      if nargin < 1, n = 21; end;
      if n==21, nfolds = -1; else   nfolds =5; end
      %[xtrain, ytrain, xtest, ytest] = makePolyData(n);
      [xtrain, ytrain, xtest, ytest] = polyDataMake('sampling', 'thibaux', 'n', n);
      deg = 14;
      m = linregDist;
      m.transformer =  chainTransformer({rescaleTransformer,polyBasisTransformer(deg)});
      lambdas = logspace(-10,1.2,15);
  
      for i=1:length(lambdas)
        lambda = lambdas(i);
        m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
        testMse(i) = mean(squaredErr(m, xtest, ytest));
        trainMse(i) = mean(squaredErr(m, xtrain, ytrain));
        [CVmeanMse(i), CVstdErrMse(i)] = cvScore(m, xtrain, ytrain, ...
          'objective', 'squaredErr', 'nfolds', nfolds);
        nparams(i) = length(m.w);
      end

      figure;
      hold on
      ndx = log(lambdas);
      plot(ndx, CVmeanMse, 'ko-', 'linewidth', 2, 'markersize', 12);
      plot(ndx, trainMse, 'bs:', 'linewidth', 2, 'markersize', 12);
      plot(ndx, testMse, 'rx-', 'linewidth', 2, 'markersize', 12);
      legend(sprintf('CV(%d)', nfolds), 'train', 'test')
      errorbar(ndx, CVmeanMse, CVstdErrMse, 'k');
      xlabel('log(lambda)')
      ylabel('mse')
      title(sprintf('poly degree %d, ntrain  %d', deg, n))

      df = dofRidge(m, xtrain, lambdas);
      ylim = get(gca, 'ylim');
      % illustrate 1 SE rule
      [bestCV bestCVndx] = min(CVmeanMse);
      % vertical line at bestCVndx
      h=line([ndx(bestCVndx) ndx(bestCVndx)], ylim); set(h, 'color', 'k');
      % horizontal line at height of bestCV
      h=line([min(ndx) max(ndx)], [bestCV bestCV]); set(h,'color','k');
      % vertical line at bestCVndx1SE
      idx_opt = oneStdErrorRule(CVmeanMse, CVstdErrMse, df);
      h=line([ndx(idx_opt) ndx(idx_opt)], ylim); set(h, 'color', 'k');
      set(gca,'ylim',[0 10]);
    end
    
    
    function demoPolyFitRidgeBIC(n)
      if nargin < 1, n = 100; end;
      if n==21, nfolds = -1; else   nfolds =5; end
      %[xtrain, ytrain, xtest, ytest] = makePolyData(n);
      [xtrain, ytrain, xtest, ytest] = polyDataMake('sampling', 'thibaux', 'n', n);
      deg = 14;
      m = linregDist();
      m.transformer =  chainTransformer({rescaleTransformer, ...
        polyBasisTransformer(deg)});
      lambdas = logspace(-10,1.2,15);
      for i=1:length(lambdas)
        lambda = lambdas(i);
        m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
        testLogprob(i) = mean(logprob(m, xtest, ytest));
        trainLogprob(i) = mean(logprob(m, xtrain, ytrain));
        [CVmeanLogprob(i), CVstdErrLogprob(i)] = cvScore(m, xtrain, ytrain, ...
          'objective', 'logprob', 'nfolds', nfolds);
        bic(i) = bicScore(m, xtrain, ytrain, lambda);
        aic(i) = aicScore(m, xtrain, ytrain, lambda);
        nparams(i) = length(m.w);
      end
      figure;
      hold on
      ndx = log(lambdas);
      n = size(xtrain,1);
      plot(ndx, CVmeanLogprob, 'ko-', 'linewidth', 2, 'markersize', 12);
      plot(ndx, trainLogprob, 'bs:', 'linewidth', 2, 'markersize', 12);
      plot(ndx, testLogprob, 'rx-', 'linewidth', 2, 'markersize', 12);
      plot(ndx, bic/n, 'g-^', 'linewidth', 2, 'markersize', 12);
      plot(ndx, aic/n, 'm-v', 'linewidth', 2, 'markersize', 12);
      legend(sprintf('CV(%d)', nfolds), 'train', 'test', 'bic', 'aic')
      errorbar(ndx, CVmeanLogprob, CVstdErrLogprob, 'k');
      xlabel('log(lambda)')
      ylabel('1/n * log p(D)')
      title(sprintf('poly degree %d, ntrain  %d', deg, n))

      % draw vertical line at best value
      ylim = get(gca, 'ylim');
      best = [argmax(CVmeanLogprob) argmax(bic) argmax(aic)];
      colors = {'k','g','m'};
      for k=1:3
        xjitter = ndx(best(k))+0.5*randn;
        h=line([xjitter xjitter], ylim);
        set(h, 'color', colors{k}, 'linewidth', 2);
      end
      if n==21, set(gca,'ylim',[-7 0]); end
    end

    function demoRbf()
      %[xtrain, ytrain, xtest, ytest] = makePolyData;
      [xtrain, ytrain, xtest, ytest] = polyDataMake('sampling','thibaux');
      lambda = 0.001; % just for numerical stability
      sigmas = [0.05 0.5 50];
      K = 10; % num centers
      for i=1:length(sigmas)
        sigma = sigmas(i);
        T = chainTransformer({rescaleTransformer, rbfBasisTransformer(K,sigma)});
        m  = linregDist('transformer', T);
        m = fit(m, 'X', xtrain, 'y', ytrain, 'lambda', lambda);
        ypred = mean(predict(m, xtest));
        figure(1);clf
        scatter(xtrain,ytrain,'b','filled');
        hold on;
        plot(xtest, ypred, 'k', 'linewidth', 3);
        title(sprintf('RBF, sigma %f', sigma))
     
        % visualize the kernel centers
        [Ktrain,T] = train(T, xtrain);
        Ktest = test(T, xtest);
        figure(2);clf; hold on
        for j=1:K
          plot(xtest, Ktest(:,j));
        end
        title(sprintf('RBF, sigma %f', sigma))
        
        figure(3);clf;
        imagesc(Ktrain); colormap('gray')
        title(sprintf('RBF, sigma %f', sigma))
        %pause
      end
    end

    function demoBasisSimple()
      linregDist.helperBasis(...
        'deg', 2, 'basis', 'quad', 'sampling', 'sparse', ...
        'plotErrorBars', true, 'prior', 'mvn');
    end
    
    function demoBasis(doPrint)
      if nargin < 1, doPrint = false; end
      folder = 'C:/kmurphy/PML/pdfFigures';
      degs = [2 3];
      basis = {'quad', 'rbf'};
      sampling = {'sparse', 'dense'};
      for i=1:length(degs)
        for j=1:length(basis)
          for k=1:length(sampling)
            linregDist.helperBasis(...
              'deg', degs(i), 'basis', basis{j}, 'sampling', sampling{k}, ...
              'plotErrorBars', true); 
          end
        end
      end
    end
       
    function demoBasisDense()
      degs = [2 3];
      basis = {'quad', 'rbf'};
      sampling = {'dense'};
      for i=1:length(degs)
        for j=1:length(basis)
          for k=1:length(sampling)
            linregDist.helperBasis(...
              'deg', degs(i), 'basis', basis{j}, 'sampling', sampling{k}, ...
              'plotErrorBars', false, 'plotSamples', false);
          end
        end
      end
    end
    
    function demoBasisSparse()
      degs = [2];
      basis = {'quad', 'rbf'};
      %basis = {'rbf'};
      sampling = {'sparse'};
      for i=1:length(degs)
        for j=1:length(basis)
          for k=1:length(sampling)
            linregDist.helperBasis(...
              'deg', degs(i), 'basis', basis{j}, 'sampling', sampling{k}, ...
              'plotErrorBars', true, 'plotSamples', true);
          end
        end
      end
    end
    
    function helperBasis(varargin)
      [sampling, deg, basis, plotErrorBars, plotBasis, prior, plotSamples] = ...
        process_options(varargin, ...
        'sampling', 'sparse', 'deg', 3, 'basis', 'rbf', ...
        'plotErrorBars', true, 'plotBasis', true, ...
        'prior', 'mvn', 'plotSamples', true);
      
      [xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2] = polyDataMake(...
        'sampling', sampling, 'deg', deg);
      switch basis
        case 'quad'
          T = polyBasisTransformer(2);
        case 'rbf'
          T = rbfBasisTransformer(10, 1);
      end
      m = linregDist('transformer', T);  
      m = inferParams(m, 'X', xtrain, 'y', ytrain, 'prior', prior, 'sigma2', sigma2);
      ypredTrain = postPredict(m, xtrain);
      ypredTest = postPredict(m, xtest);

      figure; hold on;
      h = plot(xtest, mean(ypredTest),  'k-');
      set(h, 'linewidth', 3)
      h = plot(xtest, ytestNoisefree,'b-');
      set(h, 'linewidth', 3)
      %h = plot(xtrain, mean(ypredTrain), 'gx');
      %set(h, 'linewidth', 3, 'markersize', 12)
      h = plot(xtrain, ytrain, 'ro');
      set(h, 'linewidth', 3, 'markersize', 12)
      grid off
      legend('prediction', 'truth', 'training data', ...
        'location', 'northwest')
      if plotErrorBars
        NN = length(xtest);
        ndx = 1:3:NN;
        sigma = sqrt(var(ypredTest));
        mu = mean(ypredTest);
        h=errorbar(xtest(ndx), mu(ndx), 2*sigma(ndx));
        set(h, 'color', 'k');
      end
      title(sprintf('truth = degree %d, basis = %s, n=%d, prior %s', ...
        deg, basis, length(xtrain), prior));
  
      % superimpose basis fns
      if strcmp(basis, 'rbf') && plotBasis
        ax = axis;
        ymin = ax(3);
        h=0.1*(ax(4)-ax(3));
        [Ktrain,T] = train(T, xtrain);
        Ktest = test(T, xtest);
        K = size(Ktest,2); % num centers
        for j=1:K
          plot(xtest, ymin + h*Ktest(:,j));
        end
      end
  
      % Plot samples from the posterior
      if ~plotSamples, return; end
      figure;clf;hold on
      plot(xtest,ytestNoisefree,'b-','linewidth',3);
      h = plot(xtrain, ytrain, 'ro');
      set(h, 'linewidth', 3, 'markersize', 12)
      grid off
      nsamples = 10;
      for s=1:nsamples
        ws = sample(m.w); 
        ms = linregDist('w', ws(:), 'sigma2', sigma2, 'transformer', T);
        [xtrainT, ms.transformer] = train(ms.transformer, xtrain);
        ypred = mean(predict(ms, xtest));
        plot(xtest, ypred, 'k-', 'linewidth', 2);
      end
      title(sprintf('truth = degree %d, basis = %s, n=%d', deg, basis, length(xtrain)))
    end

    function demoGaussVsNIG()
      figure;
      h(1)=linregDist.helperGaussVsNIG('prior', 'mvn', 'color', 'k');
      h(2)=linregDist.helperGaussVsNIG('prior', 'mvnIG', 'color', 'r');
      legend(h, 'mvn', 'mvnIG')
    end

    function hh=helperGaussVsNIG(varargin)
      [prior, col] = process_options(varargin, 'prior', 'mvn', 'color', 'k');
      [xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2] = polyDataMake(...
        'sampling', 'sparse', 'deg', 2);
      T = polyBasisTransformer(2);
      m = linregDist('transformer', T);
      m = inferParams(m, 'X', xtrain, 'y', ytrain, 'prior', prior, 'sigma2', sigma2);
      %ypredTrain = postPredict(m, xtrain);
      ypredTest = postPredict(m, xtest);

      hold on;
      h = plot(xtest, mean(ypredTest),  'k-');
      set(h, 'linewidth', 3)
      h = plot(xtest, ytestNoisefree,'b-');
      set(h, 'linewidth', 3)
      h = plot(xtrain, ytrain, 'ro');
      set(h, 'linewidth', 3, 'markersize', 12)
      grid off
      if 1
        NN = length(xtest);
        ndx = 1:3:NN;
        sigma = sqrt(var(ypredTest));
        mu = mean(ypredTest);
        hh=errorbar(xtest(ndx), mu(ndx), 2*sigma(ndx));
        set(hh, 'color', col);
      end
    end

     function hh=demoPolyFitErrorBars(varargin)
       [prior] = process_options(varargin, 'prior', 'mvnIG');
      [xtrain, ytrain, xtest, ytestNoisefree, ytestNoisy, sigma2] = polyDataMake(...
        'sampling', 'thibaux');
      degs = 1:2;
      for i=1:length(degs)
        deg = degs(i);
        T =  chainTransformer({rescaleTransformer, polyBasisTransformer(deg)});
        m = linregDist('transformer', T);
        m = inferParams(m, 'X', xtrain, 'y', ytrain, 'prior', prior, 'sigma2', sigma2);
        ypredTest = postPredict(m, xtest);
        figure;
        hold on;
        h = plot(xtest, mean(ypredTest),  'k-', 'linewidth', 3);
        %scatter(xtrain,ytrain,'r','filled');
        h = plot(xtrain,ytrain,'ro','markersize',14,'linewidth',3);
        NN = length(xtest);
        ndx = 1:20:NN;
        sigma = sqrt(var(ypredTest));
        mu = mean(ypredTest);
        [lo,hi] = credibleInterval(ypredTest);
        if strcmp(prior, 'mvn')
          % predictive distribution is a Gaussian
          assert(approxeq(2*1.96*sigma, hi-lo))
        end
        hh=errorbar(xtest(ndx), mu(ndx), mu(ndx)-lo(ndx), hi(ndx)-mu(ndx));
        %set(gca,'ylim',[-10 15]);
        set(gca,'xlim',[-1 21]);
      end
     end
    
     function demoPolyFitNoErrorBars()
      [xtrain, ytrain, xtest, ytestNoisefree, ytest] = polyDataMake('sampling','thibaux');
      degs = 1:2;
      for i=1:length(degs)
        deg = degs(i);
        m = linregDist;
        m.transformer =  chainTransformer({rescaleTransformer, polyBasisTransformer(deg)});
        m = fit(m, 'X', xtrain, 'y', ytrain);
        ypredTest = mean(predict(m, xtest));
        figure;
        scatter(xtrain,ytrain,'b','filled');
        hold on;
        plot(xtest, ypredTest, 'k', 'linewidth', 3);
        hold off
        title(sprintf('degree %d', deg))
        %set(gca,'ylim',[-10 15]);
        set(gca,'xlim',[-1 21]);
      end
    end

  end
end