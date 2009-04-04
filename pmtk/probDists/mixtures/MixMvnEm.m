classdef MixMvnEm < MixtureModelEm
% Mixture of Multivariate Normal Distributions    
    

methods

  function model = MixMvnEm(varargin)
    % model = MixMvnEm(...)
    % Create a model with default priors for MAP estimation
     [nmixtures,  ndims, transformer, verbose, nrestarts, maxIter, convTol]...
        = processArgs(varargin,...
        'nmixtures'    ,[] ,...
        'ndims',       [], ...
        'transformer'  ,[], ...
        'verbose',      true, ...
        'nrestarts',    3, ...
        'maxIter',      50, ...
        'convTol',      1e-3);
    K = nmixtures;
    mixingDistrib = DiscreteDist('nstates', K, 'prior', 'jeffreys');
    if isempty(ndims), error('must specify ndims'); end
    dist = MvnDist('ndims', ndims, 'prior','niw');
    distributions = copy(dist,K,1);
    model = MixtureModelEm(K, distributions, mixingDistrib, transformer, ...
      verbose, nrestarts, maxIter, convTol);
  end
 
  function model = initializeEM(model,X,r) %#ok
    K = nmixtures(model);
    [N, d] = size(X);
    [mu, assign] = kmeansSimple(X, K);
    alpha = 1; % Dirichlet pseudo count
    for k=1:K
      model.distributions{k}.mu = mu(:,k);
      members = find(assign==k);
      model.distributions{k}.Sigma = cov(X(members,:));
      model.mixingDistrib.T(k) = (length(members)+alpha)/(N + K*alpha);
    end
  end

  function displayProgress(model,data,loglik,iter,r)
    t = sprintf('EM restart %d iter %d, negloglik %g\n',iter,r,-loglik);
    fprintf(t);
    if(size(data,2) == 2)
        % Plot data and current model fit
      figure(1000);
      clf
      nmixtures = numel(model.distributions);
      if(nmixtures == 2)
        % shade points by responsibility
        %colors = subd(predict(model,data),'T')';
        post = conditional(model, data);
        colors = post.T; % colors(:,k) = prob belong to cluster k
        scatter(data(:,1),data(:,2),18,[colors(:,1),zeros(size(colors,1),1),colors(:,2)],'filled');
      else
        plot(data(:,1),data(:,2),'.','MarkerSize',10);
      end
      title(t);
      hold on;
      axis tight;
      mixingWeights = pmf(model.mixingDistrib);
      for k=1:nmixtures
        f = @(x)mixingWeights(k)*exp(logprob(model.distributions{k},x));
        [x1,x2] = meshgrid(min(data(:,1)):0.1:max(data(:,1)),min(data(:,2)):0.1:max(data(:,2)));
        z = f([x1(:),x2(:)]);
        contour(x1,x2,reshape(z,size(x1)));
        mu = model.distributions{k}.mu;
        plot(mu(1),mu(2),'rx','MarkerSize',15,'LineWidth',2);
      end
    end
  end % displayProgress

end % methods

end


