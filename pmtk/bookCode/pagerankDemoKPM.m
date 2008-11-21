% Run pagerank algorithm on Cleve Moler's 6 node web example

clear all
i = [ 2 6 3 4 4 5 6 1 1];
j = [ 1 1 2 2 3 3 3 4 6];
n = 6;
G  = sparse(i,j,1,n,n); % sparse n x n matrix with 1's in specified positions

c = sum(G,1);
k = find(c~=0); % non zero outdegree
D = sparse(k,k,1./c(k),n,n);
e = ones(n,1);
I = speye(n,n);
p = 0.85;

% Cleve's method - avoids z
x = (I - p*G*D)\e;
x = x/sum(x);

% direct method
z = ((1-p)*(c~=0) + (c==0))/n;
T = p*G*D + e*z;
pi = (I-T+ones(n,n))\e;

% Power method
format compact
x = e/n;
for i=1:10
  x = normalize((p*G*D)*x + e*(z*x));
  disp(x')
end

% Matrix free power method
[x,cnt] = pagerankpow(G)

load harvard500
figure;spy(G)
tic
[x,cnt] = pagerankpow(G);
toc
figure;bar(x);set(gca,'xlim',[-10 510]);set(gca,'ylim',[0 0.02])


