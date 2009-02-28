classdef DgmGaussDist < DgmDist
  % directed Gaussian graphical model
  
  properties
    %CPDs;
    % infMethod in GM class
    % G field is in parent class
  end

  
  %%  Main methods
  methods
   
     function dgm = mkRndParams(dgm, varargin)
       d = ndimensions(dgm);
       for j=1:d
           pa = parents(dgm.G, j);
           q  = length(pa);
           dgm.CPDs{j} = LinGaussCPD(randn(q,1), randn(1,1), rand(1,1));
       end
     end
   
    function [mu,Sigma,domain] = convertToMvn(dgm)
        % Koller and Friedman p233
        d = nnodes(dgm.G);
        mu = zeros(d,1);
        Sigma = zeros(d,d);
        for j=1:d
            if isa(dgm.CPDs{j}, 'ConstDist') % node was set by intervention
                b0 = 0; w = 0; sigma2 = 0;
            else
                b0 = dgm.CPDs{j}.w0;
                w = dgm.CPDs{j}.w;
                sigma2 = dgm.CPDs{j}.v;
            end
            pred = parents(dgm.G,j);
            beta = zeros(j-1,1);
            beta(pred) = w;
            mu(j) = b0 + beta'*mu(1:j-1);
            Sigma(j,j) = sigma2 + beta'*Sigma(1:j-1,1:j-1)*beta;
            s = Sigma(1:j-1,1:j-1)*beta;
            Sigma(1:j-1,j) = s;
            Sigma(j,1:j-1) = s';
        end
        %dist = MvnDist(mu,Sigma);
        domain = 1:length(mu);
    end
    
  end



end


