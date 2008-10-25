function G = precmatToGraph(Lambda)

G = Lambda;
G(abs(G) < 1e-4) = 0;
G = abs(sign(G));
G = setdiag(G,0);
