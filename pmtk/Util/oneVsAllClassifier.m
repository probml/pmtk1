function [yhat,varargout] = oneVsAllClassifier(varargin)
% This is a wrapper function that applies a binary classifier to a multiclass
% problem. For each class c, the binary classifier attempts to separate the
% examples belonging to class c from all of the rest. These positive
% classifications are then fixed, (the assocated examples removed) and the
% process repeated for the remaining classes. Note, this does not (necessarily)
% pick the class under which the data point is most probable as it is designed
% to work with non-probabilistic classifiers like support vector machines that
% only return predicted labels. 
%
% Matthew Dunham
%
% FORMAT:
%          [yhat,output2,output3,...] = oneVsAllClassifier('name1',val1,'name2',val2,...);
%
% INPUT: 
%        'binaryClassifier'     - A handle to a binary classifier @(Xtrain,ytrain,Xtest,options{:})
%
%        'Xtrain'               - The training X data, Xtrain(i,:) is the ith case
%
%        'Xtest'                - The test X data, Xtest(i,:) is the ith case
%
%        'ytrain'               - The training labels in any format, e.g. 1:K, strings etc. 
%
%        'binaryLabels'         - [-1,1] The binary labels to be used with the
%                                 binary classifier. (Some want [-1,1],
%                                 [0,1],[1,2], etc)
%
%        'options'              - A cell array of additional parameters to be passed to the
%                                 binary classifier
% OUTPUT:
%        'yhat'                 - The final predicted class labels in the same
%                                 support as the original ytrain labels. 
%
%        'varargout'            - If the binary classifier returns additional
%                                 outputs, these are collected here so that
%                                 varargout{j}{i} is the jth value returned by
%                                 the binary classifier during the ith fit. 


%% Process Input
    [binaryClassifier,Xtrain,ytrain,Xtest,binaryLabels,options] = process_options(varargin,...
        'binaryClassifier',[],'Xtrain',[],'ytrain',[],'Xtest',[],'binaryLabels',[-1,1],'options',{});
%% 
% We don't want to use 0 to represent unclassified because this is likely to be
% one of the binaryLabels.
    nul = -99;
    if(any(binaryLabels == nul))
        nul = -999;
        if(any(binaryLabels == nul))
            nul = -9999;
        end
    end
%%   setup
    [ytrain,map] = canonizeLabels(ytrain);     % convert labels to 1:K, remember mapping 
    nclasses = numel(map);
    yhat = nul*ones(size(Xtest,1),1);
    unclassified = 1:size(Xtest,1);
    
    noptional = min(max(nargout(binaryClassifier)-1,0),nargout());  % number of additional outputs from the binary classifier
    for i=1:noptional
       varargout{i} = cell(nclasses-1,1); 
    end
    for i=1:nclasses-1
        %% one vs rest class encoding
        ytr = ytrain;
        ytr(ytr == i) = binaryLabels(1);
        ytr(ytr ~= binaryLabels(1)) = binaryLabels(2);
        %% binary classification
        if(noptional == 0)
            yhat(unclassified) = binaryClassifier(Xtrain,ytr,Xtest(unclassified,:),options{:});
        else  % collect additional output in varargin{:}{i}
           extraOut = '';
           for j=1:noptional
               extraOut = [extraOut,sprintf('varargout{%d}{%d}, ',j,i)];
           end
           extraOut = extraOut(1:end-1);
           eval(['[yhat(unclassified) ',extraOut,'] = binaryClassifier(Xtrain,ytr,Xtest(unclassified,:),options{:});']); 
        end
        %% book keeping
        classified = find(yhat == binaryLabels(1));
        yhat(classified) = i; 
        unclassified = setdiff(unclassified,classified);
        Xtrain(ytrain == i,:) = [];
        ytrain(ytrain == i)   = [];
        
    end
    yhat(unclassified) = nclasses;   % only one class left
    yhat = map(yhat);                % map these back to the original support. 
    
end