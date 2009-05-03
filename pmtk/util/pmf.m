function p = pmf(p)
% Tabular Factor has a pmf method, which returns the factor as a Matlab double
% matrix. When the input p is a TabularFactor object, that function is
% dispatched. When p is already a double, this function is called and nothing
% needs to be done. 
warning('pmf not a method')
end