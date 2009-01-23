classdef VarElimInfEng < InfEng
% Sum-Product Variable Elimination    
    properties
        Tfacs;
        logZ;
        domain;
        visVars;
    end
 
    methods
        
        function eng = VarElimInfEng(varargin)
            
        end
        
        function eng = condition(eng, model, visVars, visValues)
            if(nargin < 3), visVars = [];end
            eng.Tfacs = convertToTabularFactors(model);
            eng.domain = model.domain;
            eng.visVars = visVars;
        end
        
        function [postQuery] = marginal(eng, queryVars)
            elim = mysetdiff(mysetdiff(eng.domain,queryVars),eng.visVars);
            % find a good ordering here
            postQuery = VarElimInfEng.variableElimination(eng.Tfacs,elim);
            eng.Tfacs = postQuery;
        end
        
        function [samples] = sample(eng,n)
            
        end
        
        function logZ = lognormconst(eng)
            
        end
        
    end
    
    
    methods(Static = true, Access = 'protected')
       
        function margFactor = variableElimination(factors,elimOrdering)
            
            k = numel(elimOrdering);
            for i=1:k
               factors = eliminate(elimOrdering(i),factors); 
            end
            margFactor = TabularFactor.multiplyFactors(factors);
            
            
            function newFactors = eliminate(variable,factors)
                nfacs = numel(factors);
                inscope  = false(nfacs,1);   % inscope(f) is true iff the variable is in the scope of factors{f} 
                for f=1:nfacs
                    inscope(f) = ismember(variable,factors{f}.domain);
                end
                phi = TabularFactor.multiplyFactors(factors(inscope));
                tau = marginalize(phi,mysetdiff(phi.domain,variable));
                newFactors = {factors{not(inscope)},tau};
            end
            
            
        end
        
        
    end
    
    
    
    methods(Static = true)
        
        function testClass()
             C = 1; S = 2; R = 3; W = 4;
             % test every combination to make sure VarElimInfEng returns the
             % same results as EnumInfEng
             powerset = {[],[C],[S],[R],[W],[C,S],[C,R],[C,W],[S,R],[S,W],[R,W],[C,S,R],[C,S,W],[C,R,W],[S,R,W],[C,S,R,W]};
             for i=1:numel(powerset)
                   dgmVE   = mkSprinklerDgm;
                   dgmENUM = dgmVE;
                   dgmVE.infEng = VarElimInfEng();
                   dgmENUM.infEng = EnumInfEng();
                   margVE = marginal(dgmVE,powerset{i});
                   margENUM = marginal(dgmENUM,powerset{i});
                   VE   = margVE.T
                   ENUM = margENUM.T
                   pause(2);
                   clc;
                   assert(approxeq(margVE.T,margENUM.T));
             end
         
        end
        
        
    end
    
end