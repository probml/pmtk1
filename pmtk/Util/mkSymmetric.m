function M = mkSymmetric(M)

n = size(M,1);
for i=1:n
  for j=1:n
    if M(i,j)==0, M(i,j)=M(j,i);
    end
  end
end
    
%T = triu(M);
%L = tril(M);
%M =T | T' | L | L';
%M = (M+M')/2;
