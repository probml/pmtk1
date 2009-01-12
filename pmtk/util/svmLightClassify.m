function [yhat,nsvecs] = svmLightClassify(Xtrain,ytrain,Xtest,sigma,trainOptions)
% This is a simple interface to svm-light.  
% Y must be in {-1,+1} (or 0 if the label is unknown)
% Sigma is the RBF bandwidth, (ignored if trainOptions is specified)
%
% OUTPUT:  
% yhat          is the predicted class labels corresponding to Xtest
% nsvecs        is the number of support vectors used


    if(ispc && isempty(cell2mat((strfind(winpath,'svmlight')))))
       error('This demo requires svmlight; please add it to the windows path. You can obtain svmlight from http://svmlight.joachims.org/');
    end

    T = StandardizeTransformer(false);
    Xtrain = train(T,Xtrain);
    Xtest  = test(T,Xtest);

    cleanup      = true;  % set to false to keep files
    trainFile    = 'train.svm';
    testFile     = 'test.svm' ;
    modelFile    = 'model.svm';
    resultsFile  = 'results.svm';
    trainLogFile = 'trainLog.svm';
    testLogFile  = 'testLog.svm';
       
    %-z c       classification
    %-t 2       for rbf expansion
    %-g sigma   to specify rbf bandwidth
    %-v 0       verbosity level 0-3 (0 is quiet)
    if(nargin < 5)
        trainOptions = sprintf('-z c -t 2 -g %f -v 0',sigma);
    end
    testOptions = '-v 0';
    
    svmWriteData(Xtrain,ytrain,trainFile);
    system(['svm_learn ',trainOptions,' ',trainFile,' ',modelFile,' > ',trainLogFile]);
    svmWriteData(Xtest,zeros(size(Xtest,1),1),testFile)
    system(['svm_classify ',testOptions,' ',testFile,' ',modelFile, ' ',resultsFile,' > ',testLogFile]);
    yhat = sign(dlmread(resultsFile));
   
    fid = fopen(modelFile);
    text = textscan(fid,'%s','delimiter','\n','whitespace','');
    fclose(fid);
    text = text{:};
    nsvecs = str2double(char(strtok(text(cellfun(@(str)~isempty(str),strfind(text,'number of support vectors plus 1'))))))-1;
    
    
    if(cleanup)
        delete(trainFile);
        delete(testFile);
        delete(modelFile);
        delete(resultsFile);
        delete(trainLogFile);
        delete(testLogFile);
    end
    
    
    
    
    
    
    function svmWriteData(X,y,fname)
    % Write data in svm_light format
    % X is nexamples-by-nfeatures
    % y is nexamples-by-1 and contains only {-1,0,1}
    % fname is the file name to which the data will be written.

        fid = fopen(fname,'w+');
        ylabels = num2cell(y);
        ylabels(y == -1) = {'-1'};
        ylabels(y ==  1) = {'+1'};
        ylabels(y ==  0) = {'0'};
        for i=1:size(X,1)
            fprintf(fid,'%s ',ylabels{i});
            for j=1:size(X,2)
                fprintf(fid,'%d:%f ',j,X(i,j));
            end
            fprintf(fid,'\n');
        end
        fclose(fid);
    end

    
    
    


end