function [seps, resids, hists]=seps_resids_hists_cell(cliques) 
% input: 1. cliques, a 1x|num_cliques| cell array of RIP ordered cliques 
% output: 1. seps, a 1x (num_cliques) cell array of the separators wrt the ordering 
%         of the cell array cliques. Note that by definition, the indexes j of S_j 
%         begin at 2, so below defines the first separator seps{1,1}=[]. 
%         2. resids, a 1x (num_cliques) cell array of the histories wrt the ordering 
%         of the cell array cliques. Note that by definition, the indexes j of R_j 
%         begin at 2, so below defines the first residual resids{1,1}=[]. 
%         3. hists, a 1x (num_cliques) cell array of the separators wrt the ordering 
%         of the cell array cliques.
 
% Written by Helen Armstrong


num_cliques=size(cliques,2); 
num_seps=num_cliques; 
num_non_empty_seps=num_seps-1; 
seps=cell(1, num_seps); 
resids=cell(1, num_seps); 
% NOTE i always set the first empty, as everyone 
% indexes j as 2,..., num_cliques. 
hists=cell(1, num_seps); 
hists{1,1}=cliques{1}; 
for index=2:num_cliques; 
    hists{1,index}=union(cliques{index}, hists{index-1});
    seps{1,index}=intersect(cliques{index}, hists{index-1}); 
    resids{1,index}=setdiff(cliques{index}, hists{index-1}); 
end; 
% Sj are intersection of total HISTORY and the
% new clique, not C_j and C_j-1. 
% All of output ordered min:max by matlab.
