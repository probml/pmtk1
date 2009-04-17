load votes; 
[LRerrorRate,NBerrorRate] = NBLRcmp(X,Y,'votes');
 
load car; 
[LRerrorRate,NBerrorRate] = NBLRcmp(X,Y,'car');

load soy; 
[LRerrorRate,NBerrorRate] = NBLRcmp(X,Y,'soy');

foo = load('docdata');
X = [foo.xtrain; foo.xtest];
Y = [foo.ytrain; foo.ytest];
[LRerrorRate,NBerrorRate] = NBLRcmp(X,Y,'docdata');
