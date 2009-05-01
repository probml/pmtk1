% vqDemo

seed = 0; randn('state', seed); rand('state', seed);
load clown;
A = X;
figure;
imagesc(A); colormap(gray); title('original')
if doPrintPmtk; pdfcrop; doPrintPmtkFigures('vqDemoClownOrig'); end;
[nrows ncols ncolors] = size(A);
data = reshape(A, [nrows*ncols ncolors]); % data(i,:) = rgb value for pixel i
%K = 8;
for K=[2,4]
  mu = kmeansSimple(data, K);
  
  % Apply codebook to quantize test image
  B = X; % test = train
  [nrows ncols ncolors] = size(B);
  data = reshape(B, [nrows*ncols ncolors]);
  compressed = kmeansEncode(data, mu);
  decompressed = kmeansDecode(compressed, mu);
  Qimg = reshape(decompressed, [nrows ncols ncolors]);
  figure;
  imagesc(Qimg); colormap(gray)
  title(sprintf('K=%d',K))
  if doPrintPmtk, doPrintPmtkFigures(sprintf('vqDemoClown%d', K)); end;
end
