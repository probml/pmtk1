%% Imputation for an MVN 

d = 10; seed = 0; pcMissing = 0.3;
setSeed(seed);
model = mkRndParams(MvnDist, d);
model.domain = 1:10;
n = 5;
Xfull = sample(model, n);
%missing = rand(n,d) < pcMissing;
missing = zeros(n,d);
missing(:, 1:floor(pcMissing*d)) = 1;
Xmiss = Xfull;
Xmiss(find(missing)) = NaN;
XmissImg = Xmiss;
XmissImg(find(missing)) = 0;
Ximpute = impute(model, Xmiss); % all the work happens here
figure; nr = 1; nc = 3;
subplot(nr,nc,1); imagesc(Xfull); title('full data'); colorbar
subplot(nr,nc,2); imagesc(XmissImg); title(sprintf('%3.2f pc missing', pcMissing)); colorbar
%subplot(2,2,3); imagesc(missing);
subplot(nr,nc,3); imagesc(Ximpute); title('imputed data'); colorbar
set(gcf,'position',[10 500 600 200])

