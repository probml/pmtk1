classdef LinregL1 < Linreg
%% Lasso Regression  (Single Variate Output)


    properties
        lambda;
        method;
    end
 
  
    %% Main methods
    methods
      function obj = LinregL1(varargin)
        % m = LinregL1(lambda, transformer, w, w0, sigma2, method, df, addOffset)
        % method is one of {'shooting', 'l1_ls'}
        [ obj.lambda, obj.transformer, obj.w, obj.w0, obj.sigma2, obj.method, obj.df, obj.addOffset] = ...
          processArgs(varargin,...
          '-lambda'      , 0, ...
          '-transformer', [], ...                     
          '-w'          , [], ...  
           '-w0'          , [], ... 
          '-sigma2'     , [], ...                     
          '-method', 'shooting', ...
          '-df', [], ...
          '-addOffset', true);
      end
       
     
       function df = dof(model)
         df = sum(abs(model.w) ~= 0);  % num non zeros
       end
        
        
        
    end % methods
    
    methods(Access = 'protected')
      
      function [w, output, model] = fitCore(model, XC, yC)
        switch lower(model.method)
          case 'shooting'
            w = LassoShooting(XC, yC, model.lambda, 'offsetAdded', false);
          case 'l1_ls'
            w = l1_ls(XC, yC, model.lambda, 1e-3, true);
          otherwise
            error('%s is not a supported L1 algorithm', model.method);
        end
        output = [];
      end
      
    end
    
  
end % class
