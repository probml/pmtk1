%% Imputation for an MVN 
%#testPMTK

error('this has been renamed mvnImputationDemo2')

d = 10; seed = 0; pcMissing = 0.3;
setSeed(seed);

% Given a new d-variate Normal MvnDist, populate with random parameters
model = mkRndParams(MvnDist, d);
% Set the variable conditioned = 1.  This is the same as model.conditioned = 1
%model = condition(model);
n = 5;
% Samples n observations from the model.  Returns as Xfull
Xfull = sample(model, n);
%missing = rand(n,d) < pcMissing;
missing = false(n,d);
% Sets certain observations to be missing
missing(:, 1:floor(pcMissing*d)) = true;

% Copy Xfull to Xmiss and then set observations in Xmiss to NaN
Xmiss = Xfull;
Xmiss((missing)) = NaN;

% Copy Xmiss to XmissImg and set NaN to 0
XmissImg = Xmiss;
XmissImg((missing)) = 0;

% Specify a prior distribution so that the code actually works (still need to deal with lack of convergence for mle / no prior case
model.prior = MvnInvWishartDist('mu',zeros(10,1),'Sigma',eye(10),'dof',11,'k',0.1);

% Perform imputation
Ximpute = emImpute(model, Xmiss); % all the work happens here

% Plotting stuff
figure; nr = 1; nc = 3;
subplot(nr,nc,1); imagesc(Xfull); title('full data'); colorbar
subplot(nr,nc,2); imagesc(XmissImg); title(sprintf('%3.2f pc missing', pcMissing)); colorbar
%subplot(2,2,3); imagesc(missing);
subplot(nr,nc,3); imagesc(Ximpute); title('imputed data'); colorbar
set(gcf,'position',[10 500 600 200])

