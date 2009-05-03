
%% check logreg functions are syntactically correct
n = 10; d = 3; C = 3;
X = randn(n,d );
y = sampleDiscrete((1/C)*ones(1,C), n, 1);
D = DataTable(X,y);


mL2 = LogregL2('-labelSpace', 1:C, '-lambda', 0.1);
mL2 = fit(mL2, D);
predMAPL2 = predict(mL2,X);
llL2 = logprob(mL2, D);

mL1 = LogregL1('-lambda', 0.1);
mL1 = fit(mL1, D);

