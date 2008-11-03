classdef modelEvaluator
% This is an abstract super class for model evaluators such as cvEvaluator of
% bicEvaluator.


   methods(Abstract = true)
       
       score = evaluateModel(varargin);
       
   end
end 
