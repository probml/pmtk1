function [] = outputFcn_example(x,optimValues,state,varargin)

fprintf('This is my output Function, f = %f\n',optimValues.fval);