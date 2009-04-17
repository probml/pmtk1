%% Find Ising structure from binary voting data
% Reproduce figure 16 from Banerjee, El Ghaoui, d'Aspremont, JMLR 2008

[X,party,senators,bills,nmissedVotes] = loadSenateData;
senatorLastNames = cellfuncell(@(C)C{end},cellfuncell(@(c)tokenize(c),lower(senators)));
figure; imagesc(X); colobar;
xlabel('senators'); ylabel('bills');
title('US voting records 2005--2006, -1=no, 1=yes, 0=absent')

% replace missing votes with noes (-1)
ndx = find(X==0);
X(ndx)=-1;

% FInd lambda using eqn 17
alpha = 0.05;
mu = mean(X);
sigma = sqrt(1-mu.^2); %eqn after 17
S = sigma' * sigma;
S = setdiag(S, inf);
mins = min(S(:));
[n,d] = size(X); 
%numerator = chi2inv(0.5*alpha*d^2, 1);
numerator = chi2inv(alpha/(2*d^2), 1);
lambda = sqrt(numerator) / (mins * sqrt(n))

W = isingLassoGgmHtf(X, lambda);

[labels,map] = canonizeLabels(party);
colors = {'b', 'y', 'r'}; % blue=democrat, yellow=independent, red=republicant
gg=graphLayout('adjMatrix', W, 'nodeNames', senatorLastNames, 'nodeColors', colors);
