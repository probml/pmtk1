function [traindata, testdata] = partitionDataset(data, pcTrain)
% Partition a dataset into a training set (of percentage size pcTrain) and test set
% We assume data is a structure with fields data.X and data.Y

Ndata = size(data.X, 1);
perm = randperm(Ndata);
Ntrain = floor(Ndata*pcTrain);
trainndx = perm(1:Ntrain);
testndx = perm(Ntrain+1:end);

traindata.X = data.X(trainndx, :);
traindata.Y = data.Y(trainndx);
testdata.X = data.X(testndx, :);
testdata.Y = data.Y(testndx);
