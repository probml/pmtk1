%% Plot a Mixture of Gaussians
m = MixGaussDist;
m.K = 3;
m.mixweights = [0.5 0.3 0.2];
m.mu(:,1) = [0.22 0.45]';
m.mu(:,2) = [0.5 0.5]';
m.mu(:,3) = [0.77 0.55]';
m.Sigma(:,:,1) = [0.018  0.01 ;  0.01 0.011];
m.Sigma(:,:,2) = [0.011 -0.01 ; -0.01 0.018];
m.Sigma(:,:,3) = m.Sigma(:,:,1);
xr = plotRange(m, 1);

figure; hold on;
colors = {'r', 'g', 'b'};
for k=1:3
    mk = MvnDist(m.mu(:,k), m.Sigma(:,:,k));
    [h,p]=plot(mk, 'useContour', true, 'xrange', xr, 'npoints', 200);
    set(h, 'color', colors{k});
end

figure;
h=plot(m, 'useLog', false, 'useContour', true, 'npoints', 200, 'xrange', xr);

figure;
h=plot(m, 'useLog', false, 'useContour', false, 'npoints', 200, 'xrange', xr);
brown = [0.8 0.4 0.2];
set(h,'FaceColor',brown,'EdgeColor','none');
hold on;
view([-27.5 30]);
camlight right;
lighting phong;
axis off;

X = sample(m, 1000);
figure;
h=plot(m, 'useLog', false, 'useContour', true, 'npoints', 200, 'xrange', xr);
hold on
plot(X(:,1), X(:,2), '.');
axis(xr)