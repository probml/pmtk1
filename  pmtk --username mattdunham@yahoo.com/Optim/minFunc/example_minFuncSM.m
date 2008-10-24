function [] = example_minFuncSM()

i = 1;
ds{i} = 'uci.flare1c.mat';i=i+1;
ds{i} = 'uci.flare1a.mat';i=i+1;
ds{i} = 'uci.flare1b.mat';i=i+1;
ds{i} = 'uci.primaryTumor.mat';i=i+1;
ds{i} = 'uci.lymphography.mat';i=i+1;
ds{i} = 'uci.flare2c.mat';i=i+1;
ds{i} = 'uci.flare2b.mat';i=i+1;
ds{i} = 'uci.flare2a.mat';i=i+1;
ds{i} = 'uci.iris.data';i=i+1;
ds{i} = 'uci.glass.data';i=i+1;
ds{i} = 'uci.wine.data';i=i+1;
ds{i} = 'uci.vowel.data';i=i+1;
ds{i} = 'uci.vehicle.data';i=i+1;
ds{i} = 'uci.led17.data';i=i+1;
ds{i} = 'statlog.satimage.data';i=i+1;
ds{i} = 'uci.waveform21.data';i=i+1;
ds{i} = 'statlog.dna.data';i=i+1;
ds{i} = 'uci.waveform40.data';i=i+1;
ds{i} = 'statlog.shuttle.data';i=i+1;

i = 1;
global_options.MaxIter = 250;
global_options.MaxFunEvals = 250;
%global_options.TolFun = 1e-16;
%global_options.TolX = 1e-16;
global_options.Display = 'full';

%% Line Searchers
% Minimize
% TR(i) = -1;
% i = i + 1;

% Steepest
%  TR(i) = 0;
%  options{i} = global_options;
%  options{i}.Method = 'sd';
%  i = i + 1;

% Conjugate Gradient
% TR(i) = 0;
% options{i} = global_options;
% options{i}.Method = 'cg';
% i = i + 1;

% Barzilai & Borwein
% TR(i) = 0;
%  options{i} = global_options;
%  options{i}.Method = 'bb';
%  i = i + 1;

% % Hessian-Free Newton
%  TR(i) = 0;
%  options{i} = global_options;
%  options{i}.Method = 'newton0';
%  i = i + 1;
%
% 
% % L-BFGS
TR(i) = 0;
options{i} = global_options;
options{i}.Method = 'lbfgs';
i = i + 1;

TR(i) = 0;
options{i} = global_options;
options{i}.Method = 'lbfgs';
options{i}.useComplex = 1;
i = i + 1;
%
% % Hessian-Free Newton + L-BFGS Precond
% TR(i) = 0;
% options{i} = global_options;
% options{i}.Method = 'newton0lbfgs';
% i = i + 1;
%
% % BFGS
% TR(i) = 0;
% options{i} = global_options;
% options{i}.Method = 'bfgs';
% i = i + 1;
% 
% % Newton
% TR(i) = 0;
% options{i} = global_options;
% options{i}.Method = 'newton';
% options{i}.HessianModify = 0;
% i = i + 1;




%% Trust Regioners
% TR(i) = 1;
% options{i} = global_options;
% options{i}.solver = 'cauchy';
% options{i}.Hessian = 'bfgs';
% i = i + 1;

% TR(i) = 1;
% options{i} = global_options;
% options{i}.solver = 'dogleg';
% options{i}.Hessian = 'bfgs';
% i = i + 1;

% TR(i) = 1;
% options{i} = global_options;
% options{i}.solver = 'Loo';
% options{i}.Hessian = 'bfgs';
% i = i + 1;

% TR(i) = 1;
% options{i} = global_options;
% options{i}.solver = 'schur';
% options{i}.Hessian = 'bfgs';
% i = i + 1;

% TR(i) = 1;
% options{i} = global_options;
% options{i}.solver = 'schur';
% options{i}.Hessian = 'exact';
% i = i + 1;

%%
for d = 1:length(ds)
   ds{d}
   [X,y,k,w_init] = loadSoftmax(ds{d});
   for o = 1:length(options)
      if TR(o) == -1
         [w fvals evals(1,o)] = minimize(w_init, 'SoftmaxLoss', -250,X,y,k);
         fval(1,o) = fvals(end);
      elseif TR(o) == 0
         [w fval(1,o) exitflag output] = minFunc(@SoftmaxLoss,w_init,options{o},X,y,k);
         evals(1,o) = output.funcCount;
      else
         [w fval(1,o) exitflag output] = minFuncTR(@SoftmaxLoss,w_init,options{o},X,y,k);
         evals(1,o) = output.funcCount;
      end
   end
    fval
    evals
    pause;
end
end

function [X,y_mc,k,w_init] = loadSoftmax(dataFile)
data = loadd(dataFile);
[n,d] = size(data);
X = full(data(:,1:d-1));
y = full(data(:,d));

u = unique(y);
k = length(u);
y_mc = zeros(size(y));
for i = 1:k
   y_mc(y==u(i)) = i;
end

% Standardize Columns and Add Bias
X = [ones(n,1) standardizeCols(X)];
[n,p] = size(X);

w_init = zeros(p*k,1);
end