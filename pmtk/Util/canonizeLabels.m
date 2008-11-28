function [canonized,support] = canonizeLabels(labels,support)
% Transform labels to 1:K. The size of canonized is the same as labels but every
% label is transformed to its corresponding entry in 1:K. If labels does not
% span the support, specify the support explicitly as the 2nd argument. 
%
% Examples:
%%
% str = {'yes'    'no'    'yes'    'yes'    'maybe'    'no'    'yes'  'maybe'};
%     
% canonizeLabels(str)
% ans =
%      3     2     3     3     1     2     3     1
%%
%  canonizeLabels([3,5,8,9,0,0,3,-1,2,4,36])
% ans =
%      4     6     7     8     2     2     4     1     3     5     9
%
%%
% Suppose we know the support is say 10:20 but our labels are [11:15,17,19] and
% we want 11 to be coded 1 not 2 since our support begins at 10 and similarly
% for 15, it should be 6 not 2. We can specify the actual support to achieve
% this.
%
% canonizeLabels([11:15,17,19])                - without specifying support
% ans =
%     1     2     3     4     5     6     7
% 
% canonizeLabels([11:15,17,19],10:20)          - with specifying support
% ans =
%     2     3     4     5     6     8    10
% 
    [nrows,ncols] = size(labels);
    labels = labels(:);

    if(nargin == 2)
        labels = [labels;support(:)];
    end
    
    if(ischar(labels))
        [s,j,canonized] = unique(labels,'rows');
       
    else
        [s,j,canonized] = unique(labels);
    end
    
    if(nargin == 2)
       if(~isequal(support(:),s(:)))
          error('Some of the data lies outside of the support.'); 
       end
       canonized(end:-1:end-numel(support)+1) = []; 
    end
    support = s;
    canonized = reshape(canonized,nrows,ncols);
end