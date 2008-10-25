function gmmOldFaithfulDemo
% Illustrate of EM for a GMM applied to the old faithful data set. 
% Code by Matthew Dunham

if(~exist('gmm','file')), error('This demo requires gmm.m');end
close all;
rand('twister',1);
X = dlmread('oldFaith.txt');
X = mkUnitVariance(X);
X = center(X);

keep = [1 3 5 16];    % Plot after these many iterations
iteration = 1;        % current iteration

markerSpec = {'filled','sizeData',70};
clusterLine = {'LineWidth',6};
fontSize = {'FontSize',14};

mu = [-1.8,1.5; 1.8, -1.5];             % starting points
sigma = cat(3,0.1*eye(2),0.1*eye(2));   % initial values for covariances
pi = [1 1];                             % initial mixing weights
g = gmm(mu,sigma,pi);                   % create a gmm object
g.nComponents = 2;                      % with two components
g.data = X;                             % assign the old faithful data
g.randomRestarts = 0;                   % just fit once
g.interweave(@visualizeFit,'update');   % display interim results

% plot before we begin
addColor = false;
started = false;
visualizeFit;    
addColor = true;
visualizeFit;
started = true;

g.fit();                                                 % fit using EM
placeFigures('nrows',1,'ncols',1,'depth',numel(keep)+2); % stack figures
for i=1:numel(keep)+2
   figure(i);
   pdfcrop;
   basedir = '.';
   print('-dpdf',fullfile(basedir,['gmmOldFaithfulDemo',num2str(i)]));
end


    function visualizeFit
    % This function is executed after each iteration of the the GMM fit
        if(ismember(iteration,keep)|| ~started)
            figure; hold on;
            res = 0.01;
            d = -2.5:res:2.5;
            [x1 x2] = meshgrid(d,d);
            [r c] = size(x1);
            data = [x1(:), x2(:)];
            p = g.pdfFactored(data);
            p1 = reshape(p(:,1),r,c);
            p2 = reshape(p(:,2),r,c);
            contour(x1,x2,p1,1,'r',clusterLine{:});
            contour(x1,x2,p2,1,'b',clusterLine{:});
            if(addColor)
                colors = g.posterior();
                colorInfo = [colors(:,1),zeros(size(colors,1),1),colors(:,2)];
            else
                colorInfo = [0 1 0];
            end
            scatter(X(:,1),X(:,2),18,colorInfo,markerSpec{:});
            set(gca,'XTick',[-2,0, 2],'YTick',[-2,0,2],...,
                      'box','on','LineWidth',3,fontSize{:});
            if(started)
                title(['Iteration ',num2str(iteration)],fontSize{:});
            end
        end
        if(started)
            iteration = iteration + 1;
        end
    end

end
