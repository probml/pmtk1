%% Apply K-means clustering to 2d old faithful data

X = load('faithful.txt');
figure; plot(X(:,1), X(:,2), '.', 'markersize', 10)
title('old faithful data')
grid on
setSeed(4);

K = 2;
%T = kmeansTransformer(K, 'doPlot', true);
T = KmeansTransformer('-K', K, '-doPlot', true);
[encoded, T, errhist] = train(T, X);

%figure; plot(errhist, '-o');
%xlabel('iter'); ylabel('train mse');


