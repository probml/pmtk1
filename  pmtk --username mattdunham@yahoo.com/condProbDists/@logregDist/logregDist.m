classdef multinomLogregDist < condProbDist
  
  properties
    w; 
    transformer;
    nclasses;
  end
  
  %% Main methods
  methods 
    function m = multinomLogregDist(varargin)
      [transformer,  w, C] = process_options(...
        varargin, 'transformer', [], 'w', [], 'nclasses', []);
      m.transformer = transformer;
      m.w = w;
      m.nclasses = C;
    end
    
   
     function [obj, output] = fit(obj, varargin)
       % model = fit(model, 'name1', val1, 'name2', val2, ...)
       % Arguments are
       % 'X' - X(i,:) Do NOT include a column of 1's
       % 'y'- y(i) in {1,...,K}
       % 'prior' - one of { 'none', 'L2', 'L1'}
       % 'lambda' - >= 0
       % 'method' - one of {'boundoptRelaxed', 'boundoptStepwise', any minFunc method}
       output = [];
       [X, y,  prior, lambda, method] = process_options(...
         varargin, 'X', [], 'y', [],  'prior', 'none', 'lambda', 0, ...
         'method', 'default');
       if lambda > 0 && strcmpi(prior, 'none'), prior = 'L2'; end
       offsetAdded = false;
       if ~isempty(obj.transformer)
         [X, obj.transformer] = train(obj.transformer, X);
         offsetAdded = obj.transformer.addOffset();
       end
       [n d] = size(X);
       C = obj.nclasses;
       if isempty(C), C = length(unique(y)); end
       Y1 = oneOfK(y, C);
       switch lower(prior)
         case {'l2', 'none'}
%            options = optimset('Display','none','Diagnostics','off','GradObj','on',...
%              'Hessian','on','DerivativeCheck','off');
           switch lower(method)
             case 'boundoptrelaxed'
               if(offsetAdded),warning('currently penalizes offset weight'),end
               [obj.w, output] = compileAndRun('boundOptL2overrelaxed',X, Y1, lambda);
               output.ftrace = output.ftrace(output.ftrace ~= -1);
             case 'boundoptstepwise'
               if(offsetAdded),warning('currently penalizes offset weight'),end
               [obj.w, output] = compileAndRun('boundOptL2Stepwise',X, Y1, lambda);
               output.ftrace = output.ftrace(output.ftrace ~= -1);
               otherwise
               if(strcmpi(method,'default'))
                   method = 'lbfgs';
               end
               
               options.Method = method;
               options.Display = 0;
               winit = zeros(d*(C-1),1);
               [obj.w, f, exitflag, output] = minFunc(@multinomLogregNLLGradHessL2, winit, options, X, Y1, lambda,offsetAdded);
           end
         case {'l1'}
           switch lower(method)
             case 'boundoptrelaxed'
               if(offsetAdded),warning('currently penalizes offset weight'),end
               [obj.w,output] =  compileAndRun('boundOptL1overrelaxed',X, Y1, lambda);
                output.ftrace = output.ftrace(output.ftrace ~= -1);
             case 'boundoptstepwise'
               if(offsetAdded),warning('currently penalizes offset weight'),end
               [obj.w, output] = compileAndRun('boundOptL1Stepwise',X, Y1, lambda); 
               output.ftrace = output.ftrace(output.ftrace ~= -1);
             case 'em'  
               lambda = lambda*ones(d,C-1);
               if(offsetAdded)
                   lambda(:,1) = 0;
               end
               lambda = lambda(:);
               options.verbose = false;
               [obj.w,fevals] = L1GeneralIteratedRidge(@multinomLogregNLLGradHessL2,zeros(d*(C-1),1),lambda,options,X,Y1,0,false);
               output = NaN(fevals,1);
               otherwise
                 lambda = lambda*ones(d,C-1);
                 if(offsetAdded)
                   lambda(:,1) = 0;
                 end
                 lambda = lambda(:);  
                 options.verbose = false;
                 [obj.w,fevals] = L1GeneralProjection(@multinomLogregNLLGradHessL2,zeros(d*(C-1),1),lambda,options,X,Y1,0);
                 output = NaN(fevals,1);
           end
         otherwise
           error(['unrecognized prior ' prior])
       end
     end
     
      function p = logprob(obj, X, y)
       % p(i) = log p(Y(i) | X(i,:), params), Y(i) in 1...C
       pred = predict(obj, X);
       P = pred.probs;
       Y = oneOfK(y, obj.nclasses);
       p =  sum(sum(Y.*log(P)));
     end
     
     function pred = predict(obj, X, w)
       % X(i,:) is case i
       % pred(i) = DiscreteDist(y|X(i,:))
       if nargin < 3, w = obj.w; end
       if ~isempty(obj.transformer)
         X = test(obj.transformer, X);
       end
       P = multiSigmoid(X,w);
       pred = discreteDist(P);
     end
     
  end
  
  %% Demos
  methods(Static = true)
    
    function test()
      % check functions are syntactically correct
      n = 10; d = 3; C = 4;
      X = randn(n,d );
      y = sampleDiscrete((1/C)*ones(1,C), n, 1);
      m = multinomLogregDist('nclasses', C);
      m = fit(m, 'X', X, 'y', y);
      pred = predict(obj, X);
      ll = logprob(obj, X, y);
    end
    
     function demoCrabs()
%       [Xtrain, ytrain, Xtest, ytest] = makeCrabs;
%       sigma2 = 32/5;
%       T = chainTransformer({standardizeTransformer(false), kernelTransformer('rbf', sigma2)});    
%       m = multinomLogregDist('nclasses',2, 'transformer', T);
%       lambda = 1e-3;
%       m = fit(m, 'X', Xtrain, 'y', ytrain, 'lambda', lambda, 'method', 'lbfgs');
%       %m = fit(m, 'X', Xtrain, 'y', ytrain, 'lambda', lambda, 'method', 'newton');
%       m.w(:)'
%       P = predict(m, Xtest);
%       yhat = mode(P); % 1 or 2
%       errs = find(yhat(:) ~= ytest(:));
%       nerrs = length(errs)
      %figure;plot(P.probs(:,1))
     end

    function demoOptimizer()
      setSeed(1);
      load soy; % n=307, d = 35, C = 3;
      %load car; % n=1728, d = 6, C = 3;
      methods = {'bb',  'cg', 'lbfgs', 'newton'};
      lambda = 1e-3;
      figure; hold on;
      [styles, colors, symbols] =  plotColors;
      for mi=1:length(methods)
        tic
        [m, output{mi}] = fit(multinomLogregDist, 'X', X, 'y', Y, ...
          'lambda', lambda, 'method', methods{mi});
        T = toc
        time(mi) = T;
        w{mi} = m.w;
        niter = length(output{mi}.ftrace)
        h(mi) = plot(linspace(0, T, niter), output{mi}.ftrace, styles{mi});
        legendstr{mi}  = sprintf('%s', methods{mi});
      end
      legend(legendstr)
    end
    
    
    function demoVisualizePredictive()


       n = 300; d = 2;
       setSeed(0);
       X = rand(n,d);
       Y = ones(n,1);
       Y(X(:,1) < (0.4+0.1*randn(n,1))) = 2; 
       Y(X(:,1) > (0.8 +0.05*randn(n,1))& X(:,2) > 0.8) = 2;
       Y(X(:,1) > 0.7 & X(:,1) < 0.8 & X(:,2) < 0.1) = 2;
     
      
       sigma2 = 1; lambda = 1e-3;
       T = chainTransformer({standardizeTransformer(false),kernelTransformer('rbf',sigma2)});
       model = multinomLogregDist('nclasses',2, 'transformer', T); 
       model = fit(model,'prior','l2','lambda',lambda,'X',X,'y',Y,'method','lbfgs'); 
       
       
       [X1grid, X2grid] = meshgrid(0:0.01:1,0:0.01:1);
       [nrows,ncols] = size(X1grid);
       testData = [X1grid(:),X2grid(:)];
       pred = predict(model,testData); 
       probGrid = reshape(pred.probs(:,1),nrows,ncols);
      
       
       figure;
       plot(X(Y==1,1),X(Y==1,2),'.r','MarkerSize',20); hold on;
       plot(X(Y==2,1),X(Y==2,2),'.b','MarkerSize',20);
       set(gca,'XTick',0:0.5:1,'YTick',0:0.5:1);
       title('Training Data');
       
       
       figure;
       surf(X1grid,X2grid,probGrid);
       shading interp;     
       view([0 90]);       
       colorbar
       set(gca,'XTick',0:0.5:1,'YTick',0:0.5:1);
       title('Predictive Distribution');
     

       
    end
    
    %{
    function demoMnist()
      load('mnistALL')
      % train_images: [28x28x60000 uint8]
      % test_images: [28x28x10000 uint8]
      % train_labels: [60000x1 uint8]
      % test_labels: [10000x1 uint8]
      setSeed(0);
      Ntrain = 100;
      Ntest = 1000;
      Xtrain = zeros(10, Ntrain, 784);
      ytrain = zeros(10, Ntrain);
      Xtest = zeros(10, Ntrain, 784);
      ytest = zeros(10, Ntest);
      for c=1:10
        ndx = find(mnist.train_labels==c);
        ndx = ndx(1:Ntrain);
        Xtrain(c,:,:) = double(reshape(mnist.train_images(:,:,ndx), [28*28 length(ndx)]))';
        ytrain(c,:) = c*ones(Ntrain,1);
        ndx = find(mnist.test_labels==c);
        ndx = ndx(1:Ntest);
        Xtest(c,:,:) = double(reshape(mnist.test_images(:,:,ndx), [28*28 length(ndx)]))';
        ytest(c,:) = c*ones(Ntest,1);
      end
      Xtrain = reshape(Xtrain, 10*Ntrain, 784);
      ytrain = ytrain(:);
      Xtest = reshape(Xtest, 10*Ntest, 784);
      ytest = ytest(:);
      
      m = fit(logregDist, 'X', Xtrain, 'y', ytrain, 'lambda', 1e-3, 'prior', 'L2');
      pred = predict(m, Xtest);
      
    end
    %}
  end
end