%% Generate data from a family tree - parameter values are hidden 
%#inprogress
setSeed(0);
doPrint = 0;

%{
   G1    G2
    | \/  |
    G3   G4
%}

G = zeros(4,4);
Gnodes = 1:4;
G(1,[3 4])=1; G(2,[3 4])=1;
  
% CPD for root nodes
rho = 0.5;
CPD{Gnodes(1)} = TabularCPD([(1-rho)^2, 2*rho*(1-rho), rho^2]);
CPD{Gnodes(2)} = CPD{Gnodes(1)};

% CPD for children (Mendel's laws)
% T(father, mother, child)
AA = 1; Aa = 2; aa = 3;
T = zeros(3,3,3);
T(AA, AA, :) = [1 0 0];
T(AA, Aa, :) = [0.5 0.5 0];
T(Aa, AA, :) = [0.5 0.5 0];
T(AA, aa, :) = [0 1 0];
T(aa, AA, :) = [0 1 0];
T(Aa, Aa, :) = [0.25 0.5 0.25];
T(Aa, aa, :) = [0 0.5 0.5];
T(aa, Aa, :) = [0 0.5 0.5];
T(aa, aa, :) = [0 0 1];
CPD{Gnodes(3)} = TabularCPD(T);
CPD{Gnodes(4)} = CPD{Gnodes(3)};

dgm = DgmDist(G,'CPDs', CPD);

% Sample from the prior
Nf = 50; % num families
Nc = 2; % num children in each family
N = Nc+2; % num members per family
Ng = 2; % num genes
Gs = zeros(Nf, N, Ng);
for g=1:Ng
  Gs(:,:,g) = sample(dgm, Nf); 
end

% Covariates
X = randn(Nf, 1);
XG = zeros(Nf, Ng+2, N);
for i=1:N
  XG(:,:,i) = [reshape(Gs(:,i,:)-1, Nf, Ng), X, ones(Nf,1)];
end
 
snrs = {MYSTERY }; % Signal to Noise
nus = {  MYSTERY }; % bit flip probability
srcs = {  MYSTERY  }; % which genes cause the effect?
Nmodels = length(snrs);

for m=1:Nmodels
  snr = snrs{m};
  src = srcs{m};
  nu = nus{m};
  lambda = 1;
  
  beta = 1*randn(Ng+2,1);
  %beta(src) = snr*randn(length(src),1); % make relevant coefficient large
  % Sample the on coefficients from a mixture of Gaussians
  S = length(src);
  bits = rand(S,1)>0.5;
  r = normrnd(-snr,1,S,1).*(1-bits) + normrnd(snr,1,S,1).*bits;
  beta(src) = r;
  
  Y = zeros(Nf, N);
  for i=1:N
    Y(:,i) = XG(:,:,i)*beta + randn(Nf,1)*sqrt(1/lambda);
  end

   nu1 = nu; nu2 = nu;
  %{
  obs = [(1-nu2)/2, nu2,(1-nu2)/2;...
          nu1, (1-2*nu1), nu1;...
         (1-nu2)/2, nu2,(1-nu2)/2];
  %}
  %'AA' is  unlikely to be called as 'aa' and vice versa
  % But Aa may be called as AA or aa
  obs = [1-nu2, nu2,0;...
         nu1, (1-2*nu1), nu1;...
         0, nu2,1-nu2];
       
  Ocpd = TabularCPD(obs);
  O = zeros(Nf, N, Ng);
  for g=1:Ng
    for i=1:N
      O(:,i,g) = sample(Ocpd, Gs(:,i,g), Nf);
    end
  end

  folder = 'C:\kmurphy\PML\pdfFigures';
  figure;
  srcstr = ['[' num2str(src) ']'];
  Nr = 2; Nc = Ng+1;
   for g=1:Ng
    subplot2(Nr,Nc,1,g); imagesc(Gs(:,:,g)-1);
    colorbar; title(sprintf('G%d', g));
  end
  subplot2(Nr,Nc,1,Nc); imagesc(X);
  betastr = sprintf('%3.1f, ', beta);
  colorbar; title(sprintf('beta=%s', betastr))
  for g=1:Ng
    subplot2(Nr,Nc,2,g); imagesc(O(:,:,g)-1); colorbar;
    title(sprintf('O%d, noise %3.2f', g, nu))
  end
  subplot2(Nr,Nc,2,Nc); imagesc(Y); colorbar
  title(sprintf('Y, src=%s, snr=%3.2f', srcstr, snr))
  fname = fullfile(folder, sprintf('familyTreeData%d.pdf', m));
  if doPrint, pdfcrop; print(gcf, '-dpdf', fname); end
 
  folder = 'C:\kmurphy\pmtk\pmtk\data';
  fname = fullfile(folder, sprintf('familyTreeData%d.mat', m));
  save(fname, 'O', 'X', 'Y');
  
end


