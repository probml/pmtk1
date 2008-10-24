classdef matrixDist < probDist
   % pdf's on positive definite matrices
   % This class contains code that is common to wishart and invWishart
   
  properties
    Sigma;
    dof;
  end
  
  methods
   
    function nll = negloglik(obj, X)
      % Negative log likelihood of a data set
      % For matrix distr: nll = -sum_i log p(X(:,:,i) | params)
      nll = -sum(logprob(obj, X),1) / size(X,3);
    end
    
    function d = ndims(obj)
      d = size(obj.Sigma,1);
    end
    
    function [h,p] = plot(obj, varargin)
     if ndims(obj)==1
       objS = convertToScalarDist(obj);
       [h,p] = plot(objS, varargin{:});
     else
       error('can only plot 1d')
     end
    end
   
    function plotMarginals(obj)
      figure;
      d = ndims(obj);
      nr = d; nc = d;
      for i=1:d
        subplot2(nr,nc,i,i);
        m = marginal(obj, i);
        plot(m, 'plotArgs', {'linewidth',2});
        title(sprintf('%s_%d','\sigma^2', i));
      end
      n = 1000;
      Sigmas = sample(obj, n);
      for s=1:n
        R(:,:,s) = cov2cor(Sigmas(:,:,s));
      end
      for i=1:d
        for j=i+1:d
          subplot2(nr,nc,i,j);
          [f,xi] = ksdensity(squeeze(R(i,j,:)));
          plot(xi,f, 'linewidth', 2);
          title(sprintf('%s(%d,%d)','\rho', i, j))
        end
      end
    end
    
    function xrange = plotRange(obj)
      if ndims(obj) > 1
        error('only works for 1d')
      end
      m = mode(obj);
      %xrange = [0.01 2];
      xrange = [0.01*m 5*mode(obj)];
    end
    
    function h = plotSamples2d(obj, n)
      % eg plotSamples2d(invWishartDist(5, randpd(2)), 4)
      figure;
      if ndims(obj) ~= 2
        error('only works for 2d')
      end
      [nr, nc] = nsubplots(n);
      Sigmas = sample(obj, n);
      for i=1:n
        pgauss = mvnDist([0 0], Sigmas(:,:,i));
        subplot(nr, nc,i)
        h=gaussPlot2d(pgauss.mu, pgauss.Sigma);
        %grid on
      end
    end
    

  end
    
    
end