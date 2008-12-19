%% Water Sprinkler Bayes Net Example
%%
%
%    C
%   / \
%  v  v
%  S  R
%   \/
%   v
%   W
%%
function dgm = mkSprinklerDgm()
    C = 1; S = 2; R = 3; W = 4;
    G = zeros(4,4);
    G(C,[S R]) = 1;
    G(S,W)=1;
    G(R,W)=1;
    % Specify the conditional probability tables as cell arrays
    % The left-most index toggles fastest, so entries are stored in this order:
    % (1,1,1), (2,1,1), (1,2,1), (2,2,1), etc.
    CPD{C} = TabularCPD(reshape([0.5 0.5], 2, 1)); %, [C]);
    CPD{R} = TabularCPD(reshape([0.8 0.2 0.2 0.8], 2, 2)); %, [C R]);
    CPD{S} = TabularCPD(reshape([0.5 0.9 0.5 0.1], 2, 2)); %, [C S]);
    CPD{W} = TabularCPD(reshape([1 0.1 0.1 0.01 0 0.9 0.9 0.99], 2, 2, 2)); %,[S R W]);
    dgm = DgmDist(G, 'CPDs', CPD);
end