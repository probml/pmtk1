function a = sub(b,ndx)
% lets you subscript the return value of a function directly without first
% storing the result. e.g. mean(rand(10),2)(3) is not allowed in matlab you have
% to go tmp = mean(rand(10),2); result = tmp(3); With this function, you can go
% sub(mean(rand(10),2),3). Use subc for {} indexing. 
   a = b(ndx); 
end