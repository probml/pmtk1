function imputationDemo()
%% Imputation on random data using specified model

d = 10;
helper(MvnDist('-ndims', d), d, false);
helper(DiscreteProdDist('-ndims', 10, '-nstates', 3), d, true);

end


function helper(baseModel, ndims, discrete)
  
pcMissingTrain = 0;
pcMissingTest = 0.3;
Ntrain = 1000;
Ntest = 5;

model = mkRndParamsb(baseModel);

%model = mkRndParams(MvnDist(), d);
Xtrain = sample(model, Ntrain);

if r
  % Random missing pattern
  missing = rand(n,d) < pcMissing;
else
  % Make the first 3 stripes (features) be completely missing
  missing = false(n,d);
  missing(:, 1:floor(pcMissing*d)) = true;
end

Xmiss = Xfull;
Xmiss(missing) = NaN;
XmissImg = Xmiss;
XmissImg(missing) = 0;
XhidImg = Xfull;
XhidImg(~missing) = 0;
[Ximpute,V] = impute(model, Xmiss); % all the work happens here

nr = 2; nc = 2;
figure; 
subplot(nr,nc,1); imagesc(Xfull); title('full data'); colorbar
%subplot(nr,nc,2); imagesc(missing); title('missing pattern'); colorbar
subplot(nr,nc,2); imagesc(XmissImg); title('observed data'); colorbar
subplot(nr,nc,3); imagesc(Ximpute); title('imputed mean'); colorbar
subplot(nr,nc,4); imagesc(XhidImg); title('hidden truth'); colorbar
%set(gcf,'position',[10 500 600 200])


hintonScale({Xfull}, {'-map', 'gray', '-title', 'full data'}, ...
  {Xfull, 1-missing}, {'-map', 'Jet', '-title', 'observed'}, ...
  {Ximpute, V}, {'-title', 'imputed mean'}, ...
  {Xfull, missing}, {'-title', 'hidden truth'});


end
