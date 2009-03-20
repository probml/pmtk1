%% Misconception tabular UGM  (Koller and Friemdna p98)
%%
%
%    A
%   / \
%  B  C
%   \/
%   D

% State 0 (false) = value 1, state 1 (true) = value 2
% Consider the potential on A, B
%     B=0    B=1
% A=0 30     5
% A=1 1      10
% p98 combines the values along rows, not along columns, so we must
% transpose
%#testPMTK
function ugm = mkMisconceptionUGM()
    A = 1; B = 2; C = 3; D = 4;
    G = zeros(4,4);
    G(A,[B, C]) = 1;
    G(B,D)=1;
    G(C,D)=1;
    G = mkSymmetric(G);
    factors{1} = TabularFactor(reshape([30, 5, 1, 10],2,2)', [A, B]);
    factors{2} = TabularFactor(reshape([100, 1, 1, 100],2,2)', [B, C]);
    factors{3} = TabularFactor(reshape([1, 100, 100, 1],2,2)', [C,D]);
    factors{4} = TabularFactor(reshape([100,1,1,100],2,2)', [D,A]);
    ugm =  UgmTabularDist('G', G, 'factors', factors,'nstates',[2,2,2,2]);
end