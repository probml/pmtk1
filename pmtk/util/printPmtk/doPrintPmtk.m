function state = doPrintPmtk
% Determine if we should print figures to disk or not.
% We control this by storing a single bit in the file
% printPmtk.m
% We use this rather than a global variable, since
% testPmtk uses clear all between demos.
  state = str2num(subc(getText('printPmtk'),1));
end
