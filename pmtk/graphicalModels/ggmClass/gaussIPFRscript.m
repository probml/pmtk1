% call fitConGraph from ggm package in R 

%{
  library(ggm)
  data(marks)
  S <- cov(marks) * 87 / 88
  G <- UG(~ mechanics*vectors*algebra + algebra*analysis*statistics)
  stuff <- fitConGraph(G, S , 88)
  covMat <- stuff$Shat
%}

%{
 library(ggm)
  n <- 10
  X <- matrix(sin(1:40),ncol=4)
  D <- data.frame(X)
  #G <- matrix(c(0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0),ncol=4)
  G <- UG(~ X1*X2 + X2*X3 + X3*X4 + X4*X1 );
  C <- cov(D)
  stuff <- fitConGraph(G,C,n=10)
  covMat <- stuff$Shat
%}

n = 10; d = 4;
%X = randn(n,d);
X = reshape(sin(1:40), n, d);
C = cov(X);
G = [0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0];


% UG(~ mechanics*vectors*algebra + algebra*analysis*statistics)
G = zeros(5,5);
me = 1; ve = 2; al= 3; an = 4; st = 5;
G([me,ve,al], [me,ve,al]) = 1;
G([al,an,st], [al,an,st]) = 1;
G = setdiag(G,0);

load marks; X = marks;
C = cov(X) * 87/88;

[precMat, covMat, klDiv] = gaussIPF(C, G);
full(covMat)

openR; 
evalR('C<-1'); evalR('n<-1'); evalR('G<-1'); evalR('stuff<-1') 
putRdata('C',C);
putRdata('n',n);
putRdata('G',C);
evalR('stuff <- fitConGraph(G,C,n)'); 
stuff = getRdata('stuff')
evalR('Shat <- stuff$Shat') 
covMatR = getRdata('Shat');
closeR;
