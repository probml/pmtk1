% Water sprinkler Bayes net
%   C
%  / \
% v  v
% S  R
%  \/
%  v
%  W

clear all
C = 1; S = 2; R = 3; W = 4;
false = 1; true = 2;

[G, CPD, joint, nodes] = makeSprinklerBN(true);

C = 1; S = 2; R = 3; W = 4;

fac{C} = tabularFactor([C], mysize(CPD{C}), CPD{C});
fac{R} = tabularFactor([C R], mysize(CPD{R}), CPD{R});
fac{S} = tabularFactor([C S], mysize(CPD{S}), CPD{S});
fac{W} = tabularFactor([S R W], mysize(CPD{W}), CPD{W});

J = multiplyFactors(fac{:});
joint3 = J.T;
assert(approxeq(joint, joint3));

dispjoint(joint3)

