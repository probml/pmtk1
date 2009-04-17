function [mi, pw] = NB_MI(theta)
% Compute mutual information between each word and the class label
% Input:
% theta(j,c) =  p word j occurs in class c
% We assume P(Y=1)=P(Y=2)=0.5
%
% Output:
% MI(j) = MI(Wj, Y) 
% pw(j) = P(Wj=1)

% I(Wk,Y) = KL(P(Wk,Y) || P(Wk)P(Y))
%        = sum_{w=0}^1 sum_{y=1}^2 P(w,y) log P(w,y)/(P(w) P(y))
%        = P(wk=0,y=1) log P(wk=0,y=1)/P(wk=0)P(y=1) 
%        + P(wk=0,y=2) log P(wk=0,y=2)/P(wk=0)P(y=2) 
%        + P(wk=1,y=1) log P(wk=1,y=1)/P(wk=1)P(y=1) 
%        + P(wk=1,y=2) log P(wk=1,y=2)/P(wk=1)P(y=2) 
%
% Note that P(Wk=1,Y=y) = P(Wk=1|Y=y) P(Y=y)

py=[.5 .5];
theta = theta';
pw = py*theta; % pw(k) = P(Wk=1|Y=1) P(Y=1) + P(Wk=1|Y=2) P(Y=2)

mi=py(1)*(1-theta(1,:)).*(log2((1-theta(1,:))./(1-pw)))+...
   py(2)*(1-theta(2,:)).*(log2((1-theta(2,:))./(1-pw)))+...
   py(1)*theta(1,:).*(log2(theta(1,:)./pw))+...
   py(2)*theta(2,:).*(log2(theta(2,:)./pw));
