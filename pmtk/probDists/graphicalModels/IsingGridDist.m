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
           
    end
    
    
end