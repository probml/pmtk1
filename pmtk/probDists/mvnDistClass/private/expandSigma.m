function C = expandSigma(Sigma, d)
% Make Sigma be dxd
switch covType(Sigma)
  case 'full'
    C = Sigma;
  case 'diag'
    C = diag(Sigma);
  case 'spherical'
    C = Sigma*ones(d);
end
end
      
       %{
       function C = contractSigma(Sigma, covType)
         % Make Sigma be small
         error('deprecated')
         switch covType
           case 'spherical'
             assert(isdiag(Sigma))
             assert(all(diag(Sigma)==Sigma(1,1)))
             C = Sigma(1,1);
           case 'diag'
             assert(isdiag(Sigma))
             C = diag(Sigma);
           case 'full'
             C = Sigma;
         end
       end
%}