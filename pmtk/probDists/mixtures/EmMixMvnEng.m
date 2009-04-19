classdef EmMixMvnEng < EmMixModelEng
% Specialized EM methods for Mixture of Multivariate Normal Distributions       

methods

 
  function model = initializeEM(eng,model,X,r) %#ok
    K = nmixtures(model);
    [N, d] = size(X);
    [mu, assign] = kmeansSimple(X, K);
    alpha = 1; % Dirichlet pseudo count
    for k=1:K
      model.distributions{k}.mu = mu(k,:)';
      members = find(assign==k);
      model.distributions{k}.Sigma = cov(X(members,:));
      model.mixingDistrib.T(k) = (length(members)+alpha)/(N + K*alpha);
    end
    model = initPrior(model, X);
  end

  function displayProgress(eng,model,data,loglik,iter,r) %#ok
    t = sprintf('EM restart %d iter %d, negloglik %g\n',r,iter,-loglik);
    fprintf(t);
    if(size(data,2) ~= 2), return; end
    % Plot data and current model fit
    figure(1000+r);
    clf
    nmixtures = numel(model.distributions);
    if(nmixtures == 2)
      % shade points by responsibility
      %colors = subd(predict(model,data),'T')';
      post = inferLatent(model,data)
      %post = conditional(model, data);
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
  end % displayProgress

end % methods

end % class


