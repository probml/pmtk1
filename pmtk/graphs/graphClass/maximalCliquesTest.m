% Whittaker's marks example
G = zeros(5,5);
me = 1; ve = 2; al= 3; an = 4; st = 5;
G([me,ve,al], [me,ve,al]) = 1;
G([al,an,st], [al,an,st]) = 1;
G = setdiag(G,0);
clq = maximalCliques(G)

