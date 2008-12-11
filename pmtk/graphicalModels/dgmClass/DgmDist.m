classdef DgmDist < GmDist
  % directed graphical model
  
  properties
    CPDs;
    infEng;
    infMethod;
    % G field is in parent class
  end

  
  %%  Main methods
  methods
    function obj = DgmDist(G, varargin)
      if isa(G,'double'), G=Dag(G); end
      obj.G = G;
      [CPDs, infMethod]= process_options(...
        varargin, 'CPDs', [], 'infMethod',[]);
      obj.CPDs = CPDs;
      obj.infMethod = infMethod;
      if ~isempty(CPDs) && ~isempty(infMethod)
        obj = initInfEng(obj);
      end
    end

    function obj = fit(obj, X, varargin)
      % We fit each CPD separately assuming no missing data
      % and no param tying
      assert(~any(isnan(X(:))));
      [n d] = size(X);
      assert(d == ndimensions(obj));
       [clampedCPDs, interventionMask] = process_options(varargin, ...
        'clampedCPDs', false(1,d), 'interventionMask', false(n,d));
      for j=find(~clampedCPDs)
        pa = parents(obj.G, j);
        %fprintf('fit node %d\n', j);
        % Only including training cases that were not set by intervention
        ndx = find(~interventionMask(:,j));
        obj.CPDs{j} = fit(obj.CPDs{j}, 'X', X(ndx, pa), 'y', X(ndx,j));
      end
      obj = initInfEng(obj);
    end
    
    function X = sample(obj, n)
      % X(i,:) = i'th forwards (ancestral) sample
      X = sample(obj.infEng, n); % not necessary to use joint!!
    end
    
    function postQuery = marginal(obj, queryVars)
      postQuery = marginal(obj.infEng, queryVars);
    end
    
    function postQuery = predict(obj, visVars, visVals, queryVars, varargin)
      % sum_h p(Q,h|V=v)
      % Needs to be vectorized...
      [interventionVector] = process_options(varargin, ...
        'interventionVector', []);
      if ~isempty(interventionVector)
        obj = performIntervention(obj, visVars, visVals, interventionVector);
      end
      postQuery = predict(obj.infEng, visVars, visVals, queryVars);
    end
     
    function obj = performIntervention(obj, visVars, visVals, interventionVector)
      % Perform Pearl's "surgical intervention" on nodes specified by
      % intervention vector. We modify the specified CPDs and 
      % reinitialize the inference engine.
      for j=interventionVector(:)'
        ndx = (visVars==j);
        obj.CPDs{j} = ConstDist(visVals(ndx));
      end
      obj = initInfEng(obj);
    end
     
    function L = logprob(obj, X, varargin)
      % L(i) = log p(X(i,:) | params)
      [n d] = size(X);
      assert(~any(isnan(X(:))));
      [interventionMask] =  process_options(varargin, ...
        'interventionMask', false(n,d));
      L = zeros(n,d);
      for j=1:d
        pa = parents(obj.G, j);
        % Only including  cases that were not set by intervention
        ndx = find(~interventionMask(:,j));
        L(ndx,j) = logprob(obj.CPDs{j}, X(ndx, pa), X(ndx, j));
      end
      L = sum(L,2);
    end

   function L = logmarglik(obj, X, varargin)
     % L = log p(X)
      [n d] = size(X);
      assert(~any(isnan(X(:))));
      [interventionMask] =  process_options(varargin, ...
        'interventionMask', false(n,d));
      L = zeros(1,d);
      for j=1:d
        pa = parents(obj.G, j);
        % Only including  cases that were not set by intervention
        ndx = find(~interventionMask(:,j));
        L(j) = logmarglik(obj.CPDs{j}, X(ndx, pa), X(ndx, j));
      end
      L = sum(L);
    end
    
    function obj = initInfEng(obj, infMethod)
      % "Freeze" the CPDs and graph structure and compile into an engine
      if nargin < 2, infMethod = obj.infMethod; end
      if isempty(infMethod), return; end % silently return
      switch lower(infMethod)
        case 'enumeration'
          T =  dgmDiscreteToTable(obj);
          obj.infEng = EnumInfEng(T);
        case 'gauss'
          [mu, Sigma] = dgmGaussToMvn(obj);
          obj.infEng = GaussInfEng(mu, Sigma);
        otherwise
          error(['unrecognized method ' infMethod])
      end
    end
    
    function dgm = mkRndParams(dgm, varargin)
      d = ndimensions(dgm);
      [CPDtype, arity] = process_options(varargin, ...
        'CPDtype', 'TabularCPD', 'arity', 2*ones(1,d));
      for j=1:d
        pa = parents(dgm.G, j);
        switch lower(CPDtype)
          case 'lingausscpd',
            q  = length(pa);
            dgm.CPDs{j} = LinGaussCPD(randn(q,1), randn(1,1), rand(1,1));
          case 'tabularcpd'
            q  = length(pa);
            dgm.CPDs{j} = TabularCPD(mkStochastic(rand(arity([pa j]))));
          otherwise
            error(['unknown type ' CPDtype])
        end
      end
      dgm = initInfEng(dgm);
    end
    
    function T = dgmDiscreteToTable(obj)
      d = length(obj.CPDs);
      Tfacs = cell(1,d);
      for j=1:d
        if isa(obj.CPDs{j}, 'ConstDist') % node was set by intervention
          sz = mysize(obj.CPDs{j}.T);
          ssz = sz(end);
          tmp = zeros(1,ssz);
          tmp(obj.CPDs{j}.point) = 1; % delta function at set value
          Tfacs{j} = TabularFactor(tmp, j);
        else % CPDs over discrete nodes can always be converted to tables...
          dom = [parents(obj.G, j), j];
          Tfacs{j} = convertToTabularFactor(obj.CPDs{j}, dom);
        end
      end
      Tfac = TabularFactor.multiplyFactors(Tfacs);
      T = Tfac.T;
      %T = TableJointDist(Tfac.T);
    end
    
    function [mu, Sigma] = dgmGaussToMvn(dgm)
      % Koller and Friedman p233
      d = nnodes(dgm.G);
      mu = zeros(d,1);
      Sigma = zeros(d,d);
      for j=1:d
        if isa(dgm.CPDs{j}, 'ConstDist') % node was set by intervention
          b0 = 0; w = 0; sigma2 = 0;
        else
          b0 = dgm.CPDs{j}.w0;
          w = dgm.CPDs{j}.w;
          sigma2 = dgm.CPDs{j}.v;
        end
        pred = parents(dgm.G,j);
        beta = zeros(j-1,1);
        beta(pred) = w;
        mu(j) = b0 + beta'*mu(1:j-1);
        Sigma(j,j) = sigma2 + beta'*Sigma(1:j-1,1:j-1)*beta;
        s = Sigma(1:j-1,1:j-1)*beta;
        Sigma(1:j-1,j) = s;
        Sigma(j,1:j-1) = s';
      end
      %mvn = MvnDist(mu,Sigma);
    end
    
  end


  methods(Static = true)
   
      function [dgm] = mkSprinklerDgm()
        % Water sprinkler Bayes net
        %   C
        %  / \
        % v  v
        % S  R
        %  \/
        %  v
        %  W
        C = 1; S = 2; R = 3; W = 4;
        G = zeros(4,4);
        G(C,[S R]) = 1;
        G(S,W)=1;
        G(R,W)=1;
        % Specify the conditional probability tables as cell arrays
        % The left-most index toggles fastest, so entries are stored in this order:
        % (1,1,1), (2,1,1), (1,2,1), (2,2,1), etc.
        CPD{C} = TabularCPD(reshape([0.5 0.5], 2, 1)); %, [C]);
        CPD{R} = TabularCPD(reshape([0.8 0.2 0.2 0.8], 2, 2)); %, [C R]);
        CPD{S} = TabularCPD(reshape([0.5 0.9 0.5 0.1], 2, 2)); %, [C S]);
        CPD{W} = TabularCPD(reshape([1 0.1 0.1 0.01 0 0.9 0.9 0.99], 2, 2, 2)); %,[S R W]);
        dgm = DgmDist(G, 'CPDs', CPD);
      end
      
      function CooperYooInterventionDemo()
        % Verify marginal likelihood equation with interventional data
        % "Causal Discovery from a Mixture of Experimental and
        % Observational Data" Cooper & Yoo, UAI 99, sec 2.2
        G = zeros(2,2); G(1,2) = 1;
        CPD{1} = TabularCPD([0.5 0.5], 'prior', 'BDeu');
        CPD{2} = TabularCPD(mkStochastic(ones(2,2)), 'prior', 'BDeu');
        dgm = DgmDist(G, 'CPDs', CPD);
        X = [2 2; 2 1; 2 2; 1 1; 1 2; 2 2; 1 1; 2 2; 1 2; 2 1; 1 1];
        M = [0 0; 0 0; 0 0; 0 0; 0 0; 1 0; 1 0; 0 1; 0 1; 0 1; 0 1];
        M = logical(M);
        dgm = fit(dgm, X, 'interventionMask', M);
        L = exp(sum(logprob(dgm, X, 'interventionMask', M))) % plugin
        L = exp(logmarglik(dgm, X, 'interventionMask', M)) % Bayes
        assert(approxeq(L, 5.97e-7))
      end
        
      
      function dgm = RainyDayDemo()
        % V    G 
        %  \  /  \
        %   v    v
        %   R    S
        V = 1; G = 2; R = 3; S = 4;
        dag = zeros(4,4);
        dag(V,R) = 1; dag(G,[R S])=1; 
        CPD{V} = TabularCPD(normalize(ones(1,2)));
        T = zeros(2,2,2);
        T(1,1,:) = [0.6 0.4];
        T(1,2,:) = [0.3 0.7];
        T(2,1,:) = [0.2 0.8];
        T(2,2,:) = [0.1 0.9];
        CPD{R} = TabularCPD(T);
        CPD{G} = TabularCPD(normalize(ones(1,2)));
        CPD{S} = TabularCPD(mkStochastic(ones(2,2)));
        dgm = DgmDist(dag, 'CPDs', CPD);
        X = [1 1 1 1;
          1 1 0 1;
          1 0 0 0];
        % compute_counts requires data to be in {1,2,...} so we use X+1
        dgm = fit(dgm, X+1, 'clampedCPDs', [0 0 1 0]);
        delta = dgm.CPDs{V}.T(1)
        alpha = dgm.CPDs{G}.T(1)
        beta = dgm.CPDs{S}.T(2,1)
        gamma = dgm.CPDs{S}.T(1,1)
      end
      
      function sprinklerDemo()
        % Example of explaining away
        dgm = DgmDist.mkSprinklerDgm;
        dgm  = initInfEng(dgm, 'enumeration');
        false = 1; true = 2;
        C = 1; S = 2; R = 3; W = 4;

        mW = marginal(dgm, W);
        assert(approxeq(mW.T(true), 0.6471))
        mSW = marginal(dgm, [S W]);
        assert(approxeq(mSW.T(true,true), 0.2781))
        
        pSgivenW = predict(dgm, W, true, S);
        assert(approxeq(pSgivenW.T(true), 0.4298));
        pSgivenWR = predict(dgm, [W R], [true, true], S);
        assert(approxeq(pSgivenWR.T(true), 0.1945)); % explaining away
        
        % Display joint
        joint = dgmDiscreteToTable(dgm);
        lab=cellfun(@(x) {sprintf('%d ',x)}, num2cell(ind2subv([2 2 2 2],1:16),2));
        figure;
        %bar(joint.T(:))
        bar(joint(:))
        set(gca,'xtick',1:16);
        xticklabelRot(lab, 90, 10, 0.01)
        title('joint distribution of water sprinkler DGM')
      end
   
      function gaussDemo()
        % Use model from Koller & Friedman p233
        G = zeros(3,3);
        G(1,2) = 1; G(2,3)=1;
        % LinGaussCPD(w, w0, sigma2)
        CPDs{1} = LinGaussCPD([], 1, 4);
        CPDs{2} = LinGaussCPD(0.5, -5, 4);
        CPDs{3} = LinGaussCPD(-1, 4, 3);
        dgm = DgmDist(G, 'CPDs', CPDs, 'infMethod', 'gauss');
        p = predict(dgm, 2, -3.1, [1 3]);
        X = sample(dgm, 1000);
        dgm2 = DgmDist(G);
        dgm2 = mkRndParams(dgm2, 'CPDtype', 'LinGaussCPD');
        dgm2 = fit(dgm2, X);
        for j=1:3
          fprintf('node %d, orig: w %5.3f, w0 %5.3f, v %5.3f, est w %5.3f, w0 %5.3f, v %5.3f\n',...
            j, dgm.CPDs{j}.w, dgm.CPDs{j}.w0, dgm.CPDs{j}.v,... 
            dgm2.CPDs{j}.w, dgm2.CPDs{j}.w0, dgm2.CPDs{j}.v);
        end
        Xtest = sample(dgm, 100);
        [n d] = size(Xtest);
        M = rand(n,d)>0.8;
        L = sum(logprob(dgm, Xtest, 'interventionMask', M))
      end
  end

end


