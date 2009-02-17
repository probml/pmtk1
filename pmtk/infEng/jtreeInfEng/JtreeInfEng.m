classdef JtreeInfEng < InfEng
% Performs calibration using sum-product message passing on the clique tree
% induced by the client model. 
    
   properties
        factors;                % the original tabular factors - unaltered
        domain;                 % the entire domain of the client model
        nstates;                % the number of states in each variable of the domain. 
        cliques;                % a cell array of TabularFactors, representing the cliques
        sepsets;                % sepsets{i,j} = cliques{i}.domain intersect cliques{j}.domain (symmetric)
        iscalibrated = false;   % true iff the clique tree is calibrated so that each clique represents the unnormalized joint over the variables in its scope. 
        messages;               % a 2D cell array s.t. messages{i,j} = the message passed from clique i to clique j. Each message is a TabularFactor.
        cliqueLookup;           % cliqueLookup(i,j) = true iff variable i is in the scope of clique j
        cliqueTree;             % The clique tree as an adjacency matrix
        cliqueScope;            % a cell array s.t. cliqueScope{i} = the scope,(domain) of the ith clique.
   end
   
   
   methods
       
       function eng = JtreeInfEng()
           eng; %#ok
       end
       
       function eng = condition(eng,model,visVars,visVals)
           if eng.iscalibrated && (nargin < 3 || isempty(visVars))
               return; % nothing to do
           end
           
           if eng.iscalibrated && nargin == 4
               eng = recalibrate(eng,visVars,visValues); % recalibrate based on new evidence. 
               return;
           end
           
           if(nargin < 4), visVars = []; visVals = {}; end
           [eng.factors,eng.nstates] = convertToTabularFactors(model,visVars,visVals);
           eng.domain = model.domain;
           if isempty(eng.cliques)
               eng = setupCliqueTree(eng,model.G);  
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
        
        
        function eng = setupCliqueTree(eng,G)
           
            nfactors = numel(eng.factors);
            ncliques = buildCliqueTree();
            addFactorsToCliques();
            constructSeparatingSets();
             
            function ncliques =  buildCliqueTree()
                if G.directed, G = moralize(G);  end
                initialGraph = G.adjMat;            % graph of the client model
                for f=1:nfactors
                    dom = eng.factors{f}.domain;
                    initialGraph(dom,dom) = 1;      % connect up factors whose scope overlap to ensure they will live in the same cliques. 
                end
                initialGraph = setdiag(initialGraph,0);
                treeObj = Jtree(initialGraph);      % The jtree class triangulates and forms a cluster tree satisfying RIP
                eng.cliqueTree  = treeObj.adjMat;   % the adjacency matrix of the clique tree
                eng.cliqueScope = treeObj.cliques;  % the scope of each clique
                ncliques = numel(eng.cliqueScope);
                eng.cliqueLookup = false(numel(eng.domain),ncliques);
                for c=1:ncliques
                    eng.cliqueLookup(eng.cliqueScope{c},c) = true; % cliqueLookup(v,c) = true iff variable v is in the scope of clique c
                end
            end
            
            function addFactorsToCliques()
                factorLookup = false(nfactors,ncliques);
                for f=1:nfactors
                    candidateCliques = all(eng.cliqueLookup(eng.factors{f}.domain,:),1);
                    c = minidx(cellfun(@(x)numel(x),eng.cliqueScope(candidateCliques))); % add the factor to the smallest accommodating clique
                    factorLookup(f,sub(find(candidateCliques),c)) = true;
                end
                eng.cliques = cell(ncliques,1);
                for c=1:ncliques
                    scope = eng.cliqueScope{c};
                    T = ones(eng.nstates(scope));
                    eng.cliques{c} = TabularFactor.multiplyFactors({TabularFactor(T,scope),eng.factors{factorLookup(:,c)}});
                end
            end
            
            function constructSeparatingSets()
                eng.sepsets = cell(ncliques);
                [is,js] = find(eng.cliqueTree);
                for k=1:numel(is)
                    i = is(k); j = js(k);
                    eng.sepsets{i,j} = myintersect(eng.cliques{i}.domain,eng.cliques{j}.domain);
                    eng.sepsets{j,i} = eng.sepsets{i,j};
                end
            end
            
        end % end of setupCliqueTree method
        
        
        function eng = recalibrate(eng,visVars, visVals)
           error('recalibration of an already calibrated tree based on new evidence is not yet implemented'); 
        end
        
        
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
            jtree = triu(eng.cliqueTree); 
            % We treat the clique tree as directed here to implicitly define a
            % topological ordering of the nodes - traversal will queue based,
            % (i.e. breadth first search), starting with the leaves. The 'queue'
            % is the readyToSend logical vector defined below. 
           
            ncliques = numel(eng.cliques);
            eng.messages = cell(ncliques);            % we store all of the messages to allow for faster recalibration given new evidence. 
            rm = @(c)c(cellfun(@(x)~isempty(x),c));   % simple function to remove empty entries in a cell array
            allexcept = @(x)[1:x-1,(x+1):ncliques];   % returns the same answer as setdiff(1:ncliques,x)
            root = sub(1:ncliques,not(sum(jtree,1))); % nodes with no parents (inwards links in the directed clique tree)
            assert(numel(root) == 1);                 % otherwise its not a directed tree!
            readyToSend = false(1,ncliques);          % readyToSend(c) = true iff clique c is ready to send its message to its parent in the upwards pass or its children in the downwards pass
            readyToSend(not(sum(jtree,2))) = true;    % leaves are ready to send
            
            % upwards pass
            while not(readyToSend(root))    
                current = sub(sub(1:ncliques,readyToSend),1); % this is the first ready clique in the queue
                parent = parents(jtree,current); assert(numel(parent) == 1);     
                psi = TabularFactor.multiplyFactors(rm({eng.cliques{current},eng.messages{allexcept(parent),current}}));
                eng.messages{current,parent} = marginalize(psi,eng.sepsets{current,parent});
                readyToSend(current) = false;
                % parent ready to send if there are messages from all of its children
                readyToSend(parent)  = all(cellfun(@(x)~isempty(x),eng.messages(children(jtree,parent),parent))); 
            end
            assert(sum(readyToSend) == 1) % only root left to send
            
            % downwards pass
            while(any(readyToSend))
                current = sub(sub(1:ncliques,readyToSend),1);
                C = children(jtree,current);
                for i=1:numel(C)
                    child = C(i);
                    psi = TabularFactor.multiplyFactors(rm({eng.cliques{current},eng.messages{allexcept(child),current}}));
                    eng.messages{current,child} = marginalize(psi,eng.sepsets{current,child});
                    readyToSend(child) = true;
                end
                readyToSend(current) = false;
            end
            
            % combine messages
            for c=1:ncliques
               eng.cliques{c} = TabularFactor.multiplyFactors(rm({eng.cliques{c},eng.messages{:,c}})); 
            end
            eng.iscalibrated = true;
        end % end of calibrate method
    end % end of protected methods block
end % end of class