function G = mkRndDAGRejectionSampling(N, edgeProb)

% We use rejection sampling to enforce acyclicity
done = 0;
maxIter = 1000;
iter = 1;
while ~done
  G = rand(N,N) > edgeProb;
  if isdag(graph(G))  | (iter > maxIter)
    done = 1;
  else
    iter = iter + 1;
  end
end
if iter > maxIter
  error('failed to make random DAG')
end
