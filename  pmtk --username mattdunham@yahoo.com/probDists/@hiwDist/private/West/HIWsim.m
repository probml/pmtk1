function [Omega,Sigma] = HIWsim(G,bG,DG,M)
%HIWSIM 
% Samples the HIW_G(bG,DG) distribution on a graph G on p nodes
%
p=size(DG,1);
Sigma = zeros(p,p,M); Omega=Sigma;    % arrays to save sampvar matrices
cliques = G{1}; separators = G{2}; 
numberofcliques = length(cliques);                            

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creat some working arrays that are computed only once                         
C1=inv(DG(cliques(1).ID,cliques(1).ID)/bG);  
c1=cliques(1).ID; UN = c1'; 
for i=2:numberofcliques
    sid = separators(i).ID;  DSi{i}=inv(DG(sid,sid));  
    cid = cliques(i).ID;     dif = setdiff(cid,UN); UN = union(cid',UN);
    sizedif = size(dif,2);   
    DRS{i} = DG(dif,dif)-DG(dif,sid)*DSi{i}*DG(sid,dif); DRS{i}=(DRS{i}+DRS{i}')/2;
    mU{i}  = DG(dif,sid)*DSi{i}; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now, MC Sampling                                                             
for j = 1:M  
   UN = c1'; 
   Sigmaj=zeros(p,p); 
   Sigmaj(c1,c1)=iwishrnd(DG(cliques(1).ID,cliques(1).ID),bG+cliques(1).dim-1);  % sample variance mx on first component
   for i=2:numberofcliques                                                       % visit components and separators in turn
      dif = setdiff(cliques(i).ID,UN); UN = union(cliques(i).ID',UN);
      sizedif = size(dif,2); sid = separators(i).ID;
      SigRS = iwishrnd(DRS{i},bG+cliques(i).dim-1);
      Ui    = rMNorm(reshape(mU{i}',1,[]),kron(SigRS,DSi{i}),1);
      Sigmaj(dif,sid) = reshape(Ui,[],sizedif)'*Sigmaj(sid,sid); Sigmaj(sid,dif) = Sigmaj(dif,sid)';
      Sigmaj(dif,dif) = SigRS + Sigmaj(dif,sid)*inv(Sigmaj(sid,sid))*Sigmaj(sid,dif);
   end
   % Next, completion operation for sampled variance matrix
   H = c1;
   for i = 2:numberofcliques
       dif = setdiff(cliques(i).ID,H); sid = separators(i).ID;
       h = setdiff(H,sid); 
       Sigmaj(dif,h) = Sigmaj(dif,sid)*inv(Sigmaj(sid,sid))*Sigmaj(sid,h);
       Sigmaj(h,dif) = Sigmaj(dif,h)';
       H=union(H,cliques(i).ID); 
   end
   Sigma(:,:,j)=Sigmaj; 
   % Next, computing the corresponding sampled precision matrix
   Caux = zeros(p,p,numberofcliques); Saux = Caux;
   cid = cliques(1).ID; Caux(cid,cid,1) = inv(Sigmaj(cid,cid));
   for i = 2:numberofcliques
      cid = cliques(i).ID;      Caux(cid,cid,i) = inv(Sigmaj(cid,cid));
      sid = separators(i).ID;   Saux(sid,sid,i) = inv(Sigmaj(sid,sid));
   end
   Saux(:,:,1)=[]; % since we have separators indexed 2 up ...
   Omega(:,:,j) = sum(Caux,3) - sum(Saux,3);
end
% End of sampling                                                         
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

