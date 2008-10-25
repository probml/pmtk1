function G = mkAdjMat(varargin)
% Create a (possibly random) graph structure
% AdjMat = mkGraph('name1', val1, 'name2', val2, ...)

[type, edgeProb, nnodes, nrows, ncols, connectivity, wrapAround, maxFanIn] = process_options(...
    varargin, 'type', [], 'edgeProb', 0.1, 'nnodes', [], ...
    'nrows', [], 'ncols', [], 'connectivity', 4, 'wrapAround', false, 'maxFanIn', []);

if isempty(maxFanIn), maxFanIn = nnodes; end

switch type
 case 'rndUG',
  G = rand(nnodes, nnodes) < edgeProb;
  G = mkSymmetric(G);
 case 'chainDir',
  % G(t,t+1) = 1 for all t<T
  G = diag(ones(1,nnodes-1),1);
 case 'chainUndir',
  G = diag(ones(1,T-1),1) + diag(ones(1,T-1),-1);
 case 'loop'
  G = tridiag(ones(nnodes, nnodes));
  G(1,nnodes) = 1; G(nnodes, 1) = 1;
 case 'lattice2D',
  if wrapAround
    assert(connectivity==4);
    if nnodes > 100
      fprintf('warning: slow!\n');
    end
    G = mk2DLatticeWrap(nrows, ncols, wrapAround);
  else
    G = mk2DLatticeNoWrap(nrows, ncols, connectivity);
  end
 otherwise
  error(['unknown type ' type])
end




