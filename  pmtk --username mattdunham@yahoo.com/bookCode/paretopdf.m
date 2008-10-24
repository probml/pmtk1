function p = pareto_pdf(x, m, K)

p = K*m^K ./ (x.^(K+1));
ndx = find(x < m);
p(ndx) = 0;
