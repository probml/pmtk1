classdef IsingModel < UgmDist 
    % Non-homogeneous Ising model 
    % p(x|theta) = 1/Z  exp[sum_i vi x(i)  +
    %    + sum_{i=1}^{d-1} sum_{j=i+1}^d w(i,j) x(i) x(j) ]
    % for xi in [-1,+1]
    properties
        W; % edge strengths (symmetric)
        v; % local potentials
    end
    
    %%  Main methods
    methods
        function obj = IsingModel(W,v)
          if nargin == 0, W = []; v  = []; end
          obj.W = W; obj.v = v;
        end
            
        function model = fitStructure(model, data)
          % Learn the graph structure using L1 penalized GGM method of
          % Banerjee
        end
    end
    
    
end