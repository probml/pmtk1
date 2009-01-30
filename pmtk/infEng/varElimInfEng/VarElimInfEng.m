classdef VarElimInfEng < InfEng
% Sum-Product Variable Elimination 
    properties
        Tfac;
        domain;
        visVars;
        ordering;
    end
 
    methods
      
        function eng = condition(eng, model, visVars, visVals)    
            if(nargin < 4), visVars = []; visVals = {};end
            [eng.Tfac,nstates] = convertToTabularFactors(model,visVars,visVals);
            if isempty(eng.ordering)
                if(model.G.directed)
                    moralGraph = moralize(model.G);
                else
                    moralGraph = model.G;
                end
                eng.ordering = best_first_elim_order(moralGraph.adjMat,nstates);
            end
            eng.domain = model.domain;
            eng.visVars = visVars;
            
            % we now slice in convertToTabularFactors
%             nvis = numel(visVars);
%             if nvis > 0
%             % slice factors according to the evidence - leave unnormalized
%             % See Koller & Friedman algorithm 9.2 pg 278   
%                 for f =1:numel(eng.Tfac)
%                       include = ismember(visVars,eng.Tfac{f}.domain);
%                       eng.Tfac{f} = slice(eng.Tfac{f},visVars(include),visValues(include));
%                 end
%             end
        end
        
        function [postQuery,Z] = marginal(eng, queryVars)
        % postQuery = sum_h p(Query,h)      
            elim = mysetdiff(mysetdiff(eng.domain(eng.ordering),queryVars),eng.visVars);
            [postQuery,Z] = normalizeFactor(VarElimInfEng.variableElimination(eng.Tfac,elim));
        end
        
        function [samples] = sample(eng,n)
           Tfac = marginal(eng,eng.domain);  
           % eng.Tfac may consist of unnormalized sliced factors after a call to
           % condition. Calling marginal(eng,eng.domain) here builds the full,
           % (conditioned) normalized table.
           samples = sample(Tfac,  n);
        end
        
        function logZ = lognormconst(eng)
            [Tfac,Z] = marginal(eng,eng.domain);
            logZ = log(Z);
        end
        
    end
    
    
    methods(Static = true, Access = 'protected')
       
        function margFactor = variableElimination(factors,elimOrdering)
        % Perform sum-product variable elimination    
        % See Koller & Friedman algorithm 9.1 pg 273
            k = numel(elimOrdering);
            for i=1:k
               factors = eliminate(elimOrdering(i),factors); 
            end
            margFactor = TabularFactor.multiplyFactors(factors);
            
            
            function newFactors = eliminate(variable,factors)
            % eliminate a single variable    
                nfacs = numel(factors);
                inscope  = false(nfacs,1);   % inscope(f) is true iff the variable is in the scope of factors{f} 
                for f=1:nfacs
                    inscope(f) = any(variable==factors{f}.domain);
                end
                psi = TabularFactor.multiplyFactors(factors(inscope));
                tau = marginalize(psi,mysetdiff(psi.domain,variable)); % marginalize out the elimination variable
                newFactors = {factors{not(inscope)},tau};
            end
            
            
        end
        
        
    end
    
    
    
    methods(Static = true)
        
        function testClass()
             C = 1; S = 2; R = 3; W = 4;
             % test every combination to make sure VarElimInfEng returns the
             % same results as EnumInfEng
             powerset = {[],C,S,R,W,[C,S],[C,R],[C,W],[S,R],[S,W],[R,W],[C,S,R],[C,S,W],[C,R,W],[S,R,W],[C,S,R,W]};
             for i=1:numel(powerset)
                   dgmVE   = mkSprinklerDgm;
                   dgmENUM = dgmVE;
                   dgmVE.infEng = VarElimInfEng();
                   dgmENUM.infEng = EnumInfEng();
                   margVE = marginal(dgmVE,powerset{i});
                   margENUM = marginal(dgmENUM,powerset{i});
                   VE   = margVE.T
                   ENUM = margENUM.T
                   assert(approxeq(margVE.T,margENUM.T));
  
             end
             %%
             dgmVE = condition(dgmVE,[R,W],[1,1]);
             pSgivenRW = marginal(dgmVE,S);
             dgmENUM = condition(dgmENUM,[R,W],[1,1]);
             pSgivenRW2 = marginal(dgmENUM,S);
             assert(approxeq(pSgivenRW.T,pSgivenRW2.T));
             %%
             [dgm] = mkSprinklerDgm;
             dgm.infEng = VarElimInfEng();
             Tfac = convertToTabularFactor(dgm);
             J = Tfac.T; % CSRW
             C = 1; S = 2; R = 3; W = 4;
             dgm = condition(dgm, [C W], [1 1]);
             pSgivenCW = marginal(dgm, S);
             pSgivenCW2 = sumv(J(1,:,:,1),3) ./ sumv(J(1,:,:,1),[2 3]);
             assert(approxeq(pSgivenCW.T(:), pSgivenCW2(:)))
             X = sample(dgm, 100);
             lognormconst(dgm)
             
        end
        
        function alarmNetworkTest()
            dgm = mkAlarmNetworkDgm();
            dgm.infEng = VarElimInfEng();
            profile on;
            for i=1:37
                m = marginal(dgm,[i,mod(i,37)+1]);
                dgm2 = condition(dgm,mod(i,37)+2,2);
                m2 = marginal(dgm2,i);
            end
            profile viewer;
        end
        
        function largeMarginalTest()
             dgm = mkAlarmNetworkDgm();
             dgm.infEng = VarElimInfEng();
             profile on
             m = marginal(dgm,1:20); % 36 million entries!
             profile viewer
        end
        
       
        
        
    end
    
end