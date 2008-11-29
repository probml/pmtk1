classdef DgmDist < GmDist
  % directed graphical model
  
  properties
    CPDs;
    infEng;
  end

  
  %%  Main methods
  methods
    function obj = DgmDist(G, varargin)
      if isa(G,'double'), G=Dag(G); end
      obj.G = G;
      [CPDs, sz, infMethod]= process_options(...
        varargin, 'CPDs', [], 'sz', [], 'infMethod', []);
      obj.CPDs = CPDs;
      %obj = createInfEng(obj);
    end

    function postQuery = marginal(obj, queryVars)
      postQuery = marginal(obj.infEng, queryVars);
    end
    
    function postQuery = predict(obj, visVars, visVals, queryVars)
      postQuery = predict(obj.infEng, visVars, visVals, queryVars);
    end
      
    function obj = mkInfEng(obj, infMethod)
      switch lower(infMethod)
        case 'bruteforce'
          d = length(obj.CPDs);
          Tfacs = cell(1,d);
          for j=1:d
            Tfacs{j} = convertToTabularFactor(obj.CPDs{j});
          end
          Tfac = TabularFactor.multiplyFactors(Tfacs);
          obj.infEng = TableJointDist(Tfac.T);
        case 'jointgauss'
          [mu, Sigma] = DgmGaussToMvn(obj);
          obj.infEng = MvnDist(mu,Sigma);
        otherwise
          error(['unrecognized method ' infMethod])
      end
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
        CPD{C} = TabularCPD(reshape([0.5 0.5], 2, 1), [C]);
        CPD{R} = TabularCPD(reshape([0.8 0.2 0.2 0.8], 2, 2), [C R]);
        CPD{S} = TabularCPD(reshape([0.5 0.9 0.5 0.1], 2, 2), [C S]);
        CPD{W} = TabularCPD(reshape([1 0.1 0.1 0.01 0 0.9 0.9 0.99], 2, 2, 2),[S R W]);
        dgm = DgmDist(G, 'CPDs', CPD);
      end
      
      function sprinklerDemo()
        % Example of explaining away
        dgm = DgmDist.mkSprinklerDgm;
        dgm  = mkInfEng(dgm, 'bruteforce');
        false = 1; true = 2;
        C = 1; S = 2; R = 3; W = 4;

        mW = marginal(dgm, W)
        assert(approxeq(mW.T(true), 0.6471))
        mSW = marginal(dgm, [S W]);
        assert(approxeq(mSW.T(true,true), 0.2781))
        
        pSgivenW = predict(dgm, W, true, S);
        assert(approxeq(pSgivenW.T(true), 0.4298));
        pSgivenWR = predict(dgm, [W R], [true, true], S);
        assert(approxeq(pSgivenWR.T(true), 0.1945)); % explaining away
      end
   
  end

end


