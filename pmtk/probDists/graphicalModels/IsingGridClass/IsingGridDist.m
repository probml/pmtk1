classdef IsingGridDist < UgmDist %& GmTabularDist
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
            
        function avgX = postMean(model, visVals, varargin)
          % visVals should be an n*m matrix
          % infMethod - one of {'gibbs'}, default model.infMethod
          % infArgs - default {}
          [infMethod, infArgs] = process_options(varargin, ...
            'infMethod', model.infMethod, 'infArgs', {});
          switch infMethod
            case 'gibbs',
              avgX = gibbsIsingGrid(model.J, model.CPDs, visVals, infArgs{:});
            otherwise
              error(['unrecognized method ' model.infMethod])
          end
        end
    end
    
    
end