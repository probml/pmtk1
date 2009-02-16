function [jtree,cliques,sepsets] = jtreeByVarElim(factors,elimOrdering)
% Construct a clique tree (jtree) via the variable elimination method. See
% Koller & Friedman Section 10.1.1. 
%
% INPUT:
%
% factors         -   a cell array of TabularFactor objects as created by calling
%                     createTabularFactors(model)
%
% elimOrdering    -  a good ordering in which to eliminate the variables, you 
%                    can use best_first_elim_order() for instance. All of the 
%                    variables in the domain must be present in elimOrdering. 
%
% OUTPUT:
%
% jtree   - an adjacency matrix representing the clique tree s.t. jtree(i,j) = 1
%           iff cliques{i} is connected to cliques{j}. 
%
% cliques - the cliques of the jtree. Each clique is represented by a single
%           TabularFactor object, (i.e. the clique has already been initialized)
%
% sepsets - The clique separators s.t. 
%           sepsets{i,j} = intersect(cliques{i}.domain,cliques{j}.domain)

% BUG! Cliques don't always satisfy RIP
   

    %% Run variable elimination to build the clique tree. 
    N = numel(elimOrdering);
    jtree   = zeros(N);
    cliques = cell(N,1);
    
    activeTau = zeros(1,numel(factors)); 
    for i=1:N
        variable = elimOrdering(i);  
        inscope = cellfun(@(fac)any(variable == fac.domain),factors);
        jtree(nonzeros(activeTau(inscope)),variable) = 1;
        psi = TabularFactor.multiplyFactors(factors(inscope));
        cliques{i} = psi;
        tau = marginalize(psi,mysetdiff(psi.domain,variable));                  
        factors = {factors{not(inscope)},tau};
        activeTau = [activeTau(not(inscope)),variable];
    end
     
  
    %% Remove non-maximal cliques
   
    domains  = cellfuncell(@(c)c.domain,cliques);
    perm     = sortidx(cellfun(@(d)numel(d),domains));
    domains  = domains(perm);
    remove   = false(1,N);
    for i=1:N
       dom = domains{i};
       for j=i+1:N
           if issubset(dom,domains{j})
              remove(perm(i)) = true;
              break; 
           end
       end    
    end
 
    rmNDX = find(remove);
    for i=1:numel(rmNDX)
       r = rmNDX(i);
       P = parents(jtree,r);
       C = children(jtree,r);
%        fam = [P,C];
%        for j=1:numel(fam)
%            f = fam(j);
%            cliques{r} = TabularFactor.multiplyFactors({cliques{r},cliques{f}});
%        end
       jtree(P,C) = 1;
    end
    cliques(remove) = [];
    jtree(rmNDX,:)  = [];
    jtree(:,rmNDX)  = [];
    jtree = jtree';
    assert(~any(diag(jtree)));
    
    
    %% Construct Separating Sets
    sepsets = cell(numel(cliques));
    [is,js] = find(jtree);
    for k=1:numel(is)
       i = is(k); j = js(k);
       sepsets{i,j} = myintersect(cliques{i}.domain,cliques{j}.domain);
       sepsets{j,i} = sepsets{i,j};
    end
    
end