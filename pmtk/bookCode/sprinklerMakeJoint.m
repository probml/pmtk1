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

% Specify the conditional probability tables as cell arrays
% The left-most index toggles fastest, so entries are stored in this order:
% (1,1,1), (2,1,1), (1,2,1), (2,2,1), etc. 

CPD{C} = reshape([0.5 0.5], 2, 1);
CPD{R} = reshape([0.8 0.2 0.2 0.8], 2, 2);
CPD{S} = reshape([0.5 0.9 0.5 0.1], 2, 2);
CPD{W} = reshape([1 0.1 0.1 0.01 0 0.9 0.9 0.99], 2, 2, 2);


joint = zeros(2,2,2,2);
for c=1:2
  for r=1:2
    for s=1:2
      for w=1:2
	joint(c,s,r,w) = CPD{C}(c) * CPD{S}(c,s) * CPD{R}(c,r) * CPD{W}(s,r,w);
      end
    end
  end
end

joint2 = repmat(reshape(CPD{C}, [2 1 1 1]), [1 2 2 2]) .* ...
	 repmat(reshape(CPD{S}, [2 2 1 1]), [1 1 2 2]) .* ...
	 repmat(reshape(CPD{R}, [2 1 2 1]), [1 2 1 2]) .* ...
	 repmat(reshape(CPD{W}, [1 2 2 2]), [2 1 1 1]);
assert(approxeq(joint, joint2));

%fac{C} = tabularFactor([C], [2], [0.5 0.5]);
%fac{R} = tabularFactor([C R], [2 2], [0.8 0.2 0.2 0.8]);
%fac{S} = tabularFactor([C S], [2 2], [0.5 0.9 0.5 0.1]);
%fac{W} = tabularFactor([S R W], [2 2 2], [1 0.1 0.1 0.01 0 0.9 0.9 0.99]);

fac{C} = tabularFactor([C], mysize(CPD{C}), CPD{C});
fac{R} = tabularFactor([C R], mysize(CPD{R}), CPD{R});
fac{S} = tabularFactor([C S], mysize(CPD{S}), CPD{S});
fac{W} = tabularFactor([S R W], mysize(CPD{W}), CPD{W});

J = multiplyFactors(fac{:});
joint3 = J.T;
assert(approxeq(joint, joint3));

dispjoint(joint3)

ndx=ind2subv([2 2 2 2],1:16);
for i=1:16
  lab{i} =sprintf('%d ',ndx(i,:));
end
% fancy vectorized method:
lab=cellfun(@(x) {sprintf('%d ',x)}, num2cell(ind2subv([2 2 2 2],1:16),2))


figure(1);clf;bar(joint3(:))
set(gca,'xtick',1:16);
xticklabelRot(lab, 90, 10, 0.01)
title('joint distribution of water sprinkler DGM')
