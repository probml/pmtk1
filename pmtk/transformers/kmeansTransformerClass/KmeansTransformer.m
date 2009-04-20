classdef KmeansTransformer < Transformer
    
  properties
    K;
    codebook; % mu(k,:) is k'th center
    maxIter;
    doPlot;
  end

  methods

    function obj = KmeansTransformer(varargin)
      % KmeansTransformer(K, [maxIter], [doPlot])
      if nargin ==0; return; end
      [obj.K, obj.maxIter, obj.doPlot] = ...
        processArgs(varargin, '-K', [], '-maxIter', 100, '-doPlot', false);
    end

    function [Xdiscrete, obj, errhist] = train(obj, X)
      % Compute the code book
      % X(i,:) is i'th real-valued vector
      % Xdiscrete(i) is a number in {1,...,K}
      % errhist(t) is reconstruction error vs t
      if obj.doPlot && size(X,2)==2
        plotter = @kmeansPlotter;
      else
        plotter = [];
      end
      [obj.codebook, Xdiscrete, errhist] = ...
        kmeansSimple(X, obj.K, 'maxIter', obj.maxIter, 'progressFn', plotter);
    end

    function [Xdiscrete, Xrecon] = test(obj,X)
      % Use the codebook
      % X(i,:) is i'th real-valued vector
      % Xdiscrete(i) is a number in {1,...,K}
      % Xrecon(i,:) is reconstruction of X(i,:) using codebook
      Xdiscrete = kmeansEncode(X, obj.codebook);
      if nargout >= 2
        Xrecon = kmeansDecode(Xdiscrete, obj.codebook);
      end
    end

  end


end