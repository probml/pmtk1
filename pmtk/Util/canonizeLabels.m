function labels = canonizeLabels(labels)
% Transform labels to 1:K
%
% Example:
%%
% str = 
%     'yes'    'no'    'yes'    'yes'    'maybe'    'no'    'yes'    'maybe'
% canonizeLabels(str)
% ans =
%      3     2     3     3     1     2     3     1
%%
%  canonizeLabels([3,5,8,9,0,0,3,-1,2,4,36])
% ans =
%      4     6     7     8     2     2     4     1     3     5     9


    if(ischar(labels))
        [j,j,labels] = unique(labels,'rows');
        labels = labels';
    else
        [j,j,labels] = unique(labels);
    end
    
    
end