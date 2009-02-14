classdef IsingGridDist < GmDist
    % 2D Ising model with local evidence
    
    properties
        J; % coupling strength
        CPDs; % CPD{k} = p(y|X=k) local evidence
    end
    
    %%  Main methods
    methods
        function obj = IsingGridDist(J, CPDs)
           obj.J = J;
           obj.CPDs = CPDs;
        end
           
        function avgX = postMean(model, visVars, visVals)
          % visVars is ignored
          % visVals should be an n*m matrix
          avgX = postMean(model.infEng, model, visVars, visVals);
        end
    end
    
    
end