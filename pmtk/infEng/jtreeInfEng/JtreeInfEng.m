classdef JtreeInfEng < InfEng
% Performs calibration using sum-product message passing on the clique tree
% induced by the client model. 
    
   properties
        factors;                % the original tabular factors - unaltered
        domain;                 % the entire domain of the client model
        cliques;                % a cell array of TabularFactors, representing the cliques
        sepsets;                % sepsets{i,j} = cliques{i}.domain intersect cliques{j}.domain (symmetric)
        iscalibrated = false;   % true iff the clique tree is calibrated so that each clique represents the unnormalized joint over the variables in its scope. 
        messages;               % a 2D cell array s.t. messages{i,j} = the message passed from clique i to clique j. Each message is a TabularFactor.
        cliqueLookup;           % cliqueLookup(i,j) = true iff variable i is in the scope of clique j
        cliqueTree;             % The clique tree as an adjacency matrix
        cliqueScope;            % a cell array s.t. cliqueScope{i} = the scope,(domain) of the ith clique.
   end
   
   
   methods
       
       function eng = condition(eng,model,visVars,visVals)
           
           if eng.iscalibrated && (nargin < 3 || isemtpy(visVars))
               return; % nothing to do
           end
           if(nargin < 4)
              visVars = []; visVals = {};
           end
           [eng.factors,nstates] = convertToTabularFactors(model,visVars,visVals);
           eng.domain = model.domain;
           if isempty(eng.cliques)
               G = model.G;
               if G.directed
                  G = moralize(G); 
               end
               %ordering = best_first_elim_order(G.adjMat,nstates);
               %[eng.cliqueTree,eng.cliques,eng.sepsets] = jtreeByVarElim(eng.factors,ordering); 
               adjmat = G.adjMat;
%                 for f = 1:numel(eng.factors)
%                    dom = eng.factors{f}.domain;
%                    adjmat(dom,dom) = 1;
%                end
%                adjmat = setdiag(adjmat,0);
               
               tree = Jtree(adjmat);
               adjmat = tree.adjMat;
               eng.cliqueScope = tree.cliques;
               ncliques = numel(eng.cliqueScope);
               eng.cliqueLookup = false(numel(eng.domain),ncliques);
               for c=1:ncliques
                    eng.cliqueLookup(eng.cliqueScope{c},c) = true;
               end
               nfactors = numel(eng.factors);
               factorLookup = false(nfactors,ncliques);
               for f=1:nfactors
                    candidateCliques = all(eng.cliqueLookup(eng.factors{f}.domain,:),1);
                    c = minidx(cellfun(@(x)numel(x),eng.cliqueScope(candidateCliques)));
                    factorLookup(f,sub(find(candidateCliques),c)) = true;
               end
             
               emptyCliques = find(~any(factorLookup,1));
               adjmat = triu(mkSymmetric(adjmat));
               for i=1:numel(emptyCliques)
                   ec = emptyCliques(i);
                   P = parents(adjmat,ec);
                   C = children(adjmat,ec);
                   adjmat(P,C) = 1;
               end
               adjmat(emptyCliques,:)   = [];
               adjmat(:,emptyCliques)   = [];
               eng.cliqueScope(emptyCliques)    = [];
               eng.cliqueLookup(:,emptyCliques) = [];
               factorLookup(:,emptyCliques) = [];
               ncliques = size(eng.cliqueLookup,2);
               
               eng.cliques = cell(ncliques,1);
               for c=1:ncliques
                   eng.cliques{c} = TabularFactor.multiplyFactors({TabularFactor(1,eng.cliqueScope{c}),eng.factors{factorLookup(:,c)}});
               end
             
               eng.sepsets = cell(ncliques);
               [is,js] = find(adjmat);
               for k=1:numel(is)
                   i = is(k); j = js(k);
                   eng.sepsets{i,j} = myintersect(eng.cliques{i}.domain,eng.cliques{j}.domain);
                   eng.sepsets{j,i} = eng.sepsets{i,j};
               end
              
               eng.cliqueTree = adjmat;
               
%                ncliques = size(factorLookup,2);
%                eng.cliques   = cell(ncliques,1);
%                cliqueDomains = cell(ncliques,1);
%                for c=1:ncliques
%                   eng.cliques{c}   = TabularFactor.multiplyFactors(eng.factors(factorLookup(:,c))); 
%                   cliqueDomains{c} = eng.cliques{c}.domain;
%                end
               
               
               
               %ncliques = numel(eng.cliques);
               %eng.cliqueLookup = false(numel(eng.domain),ncliques);
%                eng.cliqueScope = cellfuncell(@(c)c.domain,eng.cliques);
%                for c=1:ncliques
%                    eng.cliqueLookup(eng.cliqueScope{c},c) = true;
%                end
           end
           eng = calibrate(eng);
       end
       
       function [postQuery,eng] = marginal(eng,queryVars)
           assert(eng.iscalibrated);
           cliqueNDX = findClique(eng,queryVars);
           if isempty(cliqueNDX)
               postQuery = outOfCliqueQuery(eng,queryVars);
           else
               postQuery = normalizeFactor(marginalize(eng.cliques{cliqueNDX},queryVars));
           end
       end
       
       function samples = sample(eng,n)
           samples = sample(normalizeFactor(TabularFactor.multiplyFactors(eng.factors)),n);
       end
       
       function logZ = lognormconst(eng)
            [Tfac,Z] = normalizeFactor(TabularFactor.multiplyFactors(eng.factors)); %#ok
            logZ = log(Z);
       end

   end
    
    
    methods(Access = 'protected')
        

        
        function postQuery = outOfCliqueQuery(eng,queryVars)
            error('out of clique query not yet implemented.');
        end
        
        function ndx = findClique(eng,queryVars)
            dom = 1:size(eng.cliqueLookup,1);
            candidates = dom(all(eng.cliqueLookup(queryVars,:),1));
            if isempty(candidates)
                ndx = []; 
            elseif numel(candidates) == 1
               ndx = candidates;
            else % return the smallest valid clique;
               ndx = candidates(minidx(cellfun(@(x)numel(x),eng.cliques(candidates))));
            end
        end
        
        function eng = calibrate(eng)
            adjmat           = eng.cliqueTree;
            %[junk,orderUp] = dfs(adjmat,1,1);         %#ok depth first search traversal induces a topological ordering of the nodes
            %orderDown   = orderUp(end:-1:1);
            ncliques = numel(eng.cliques);
            eng.messages = cell(ncliques);
            rm = @(c)c(cellfun(@(x)~isempty(x),c));
            allexcept = @(x)[1:x-1,(x+1):ncliques];
            %initialCliques = eng.cliques;
            
            root = sub(1:ncliques,not(sum(adjmat,1))); 
   
            assert(numel(root) == 1);
            readyToSend = false(1,ncliques);
            readyToSend(not(sum(adjmat,2))) = true; % leaves are ready
            % upwards pass
            while not(readyToSend(root))    
                current          = sub(sub(1:ncliques,readyToSend),1);
                parent           = parents(adjmat,current);
                assert(numel(parent) == 1);
                messagesIn       = rm(eng.messages(allexcept(parent),current));
                if isempty(messagesIn)
                   messageOut = marginalize(eng.cliques{current},eng.sepsets{current,parent}); 
                else
                   messageOut = marginalize(TabularFactor.multiplyFactors({eng.cliques{current},messagesIn{:}}),eng.sepsets{current,parent}); 
                end
                eng.messages{current,parent} = messageOut;
                readyToSend(current) = false;
                C = children(adjmat,parent);
                readyToSend(parent) = all(cellfun(@(x)~isempty(x),eng.messages(C,parent))); % parent ready to send if there are messages from all of its children
            end
            assert(sum(readyToSend) == 1 && readyToSend(root)) % only root left
%             
%             C = children(adjmat,root);
%            
%             for i=1:numel(C)
%                 child = C(i);
%                 messagesInExceptChild    = rm(eng.messages(allexcept(child),root));
%                 if isempty(messagesInExceptChild)
%                    eng.messages{root,child} = marginalize(eng.cliques{root},eng.sepsets{root,child}); 
%                 else
%                     eng.messages{root,child} = marginalize(TabularFactor.multiplyFactors({eng.cliques{root},messagesInExceptChild{:}}),eng.sepsets{root,child});
%                 end
%             end
%             readyToSend(root) = false;
%             readyToSend(C) = true;

            

            
            while(any(readyToSend))
                current  = sub(sub(1:ncliques,readyToSend),1);
                C = children(adjmat,current);
                for i=1:numel(C)
                    child = C(i);
                    parent = parents(adjmat,current);
                    %messagesIn = rm({eng.messages{parent,current},eng.messages{allexcept(current),parent}});
                    messagesIn = rm(eng.messages(parent,current));
                    if isempty(messagesIn)
                        eng.messages{current,child} = marginalize(eng.cliques{current},eng.sepsets{current,child});
                    else
                        eng.messages{current,child} = marginalize(TabularFactor.multiplyFactors({eng.cliques{current},messagesIn{:}}),eng.sepsets{current,child});
                    end
                    readyToSend(child) = true;
                end
                readyToSend(current) = false;
            end
            
            for c=1:ncliques
               eng.cliques{c} = TabularFactor.multiplyFactors(rm({eng.cliques{c},eng.messages{:,c}})); 
            end
            
            
            
%             
%             % upwards pass     
%             for k=1:numel(orderUp)-1
%                i = orderUp(k);
%                j = parents(adjmat,i);
%                assert(~isempty(j) && numel(j) < 2);  % otherwise its not a tree
%                %eng.messages{i,j} = marginalize(TabularFactor.multiplyFactors(rm({eng.cliques{i},eng.messages{:,i}})),eng.sepsets{i,j});
%                eng.messages{i,j} = marginalize(eng.cliques{i},eng.sepsets{i,j});
%                eng.cliques{j}    = TabularFactor.multiplyFactors({eng.cliques{j},eng.messages{i,j}});
%             end
%                
%             % collect
%              
% %             root = orderUp(end);
% %             C = children(adjmat,root);
% %             for j=1:numel(C)
% %                 child = C(j);
% %                 eng.messages{root,child} = marginalize(TabularFactor.multiplyFactors(rm({initialCliques{root},eng.messages{allexcept(child),root}})),eng.sepsets{root,child});
% %             end
%             
%             % downwards pass
%             
%             for d=2:numel(orderDown)
%                j = orderDown(d);
%                C = children(adjmat,j);
%                for k=1:numel(C)
%                   i = C(k); 
%                   eng.messages{j,i} = marginalize(TabularFactor.multiplyFactors(rm({initialCliques{j},eng.messages{allexcept(j),j}})),eng.sepsets{j,i}); 
%                   %eng.cliques{i} = TabularFactor.multiplyFactors({eng.cliques{i},eng.messages{j,i}}); % NOT READY TO SEND YET
%                end
%                
%             end
%             
%             for d =2:numel(orderDown)
%                 i = orderDown(d);
%                eng.cliques{i} = TabularFactor.multiplyFactors(rm({eng.cliques{i},eng.messages{orderDown(2:d-1),i}}));
%             end

            eng.iscalibrated = true;
        end
      
    end
    
    
    
end