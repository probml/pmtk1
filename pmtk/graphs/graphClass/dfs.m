function [d, pre, post, cycle, f, pred] = dfs(adj_mat, start, directed, verbose)
% DFS Perform a depth-first search of the graph starting from 'start'.
% [d, pre, post, cycle, f, pred] = dfs(adj_mat, start, directed)
% Input:
% adj_mat(i,j)=1 iff i is connected to j.
% start is the root vertex of the dfs tree; if [], all nodes are searched
% directed = 1 if the graph is directed
%
% Output:
% d(i) is the time at which node i is first discovered.
% pre is a list of the nodes in the order in which they are first encountered (opened).
% post is a list of the nodes in the order in which they are last encountered (closed).
% 'cycle' is true iff a (directed) cycle is found.
% f(i) is the time at which node i is finished.
% pred(i) is the predecessor of i in the dfs tree.
%
% If the graph is a tree, preorder is parents before children,
% and postorder is children before parents.
% For a DAG, topological order = reverse(postorder).
%
% See Cormen, Leiserson and Rivest, "An intro. to algorithms" 1994,
% p478.

if nargin < 4, verbose = false; end
 
n = length(adj_mat);

global white gray black color
white = 0; gray = 1; black = 2;
color = white*ones(1,n);

global time_stamp
time_stamp = 0;

global d f
d = zeros(1,n);
f = zeros(1,n);

global pred
pred = zeros(1,n);

global cycle
cycle = false;

global pre post
pre = [];
post = [];

if ~isempty(start)
  dfs_visit(start, adj_mat, directed, verbose);
end
for u=1:n
  if color(u)==white
    dfs_visit(u, adj_mat, directed, verbose);
  end
end

end


function dfs_visit(u, adj_mat, directed, verbose)
global white gray black color time_stamp d f pred cycle  pre post

pre = [pre u];
color(u) = gray;
time_stamp = time_stamp + 1;
d(u) = time_stamp;
if directed
  ns = children(adj_mat, u);
else
  ns = neighbors(adj_mat, u);
  ns = setdiff(ns, pred(u)); % don't go back to visit the guy who called you!
end
for v=ns(:)'
  %if verbose, fprintf('u=%d, v=%d, color(v)=%d\n', u, v, color(v)); end
  switch color(v)
    case white, % not visited v before (tree edge)
      pred(v)=u;
      dfs_visit(v, adj_mat, directed, verbose);
    case gray, % back edge - v has been visited, but is still open
      cycle = true;
      if verbose, fprintf('cycle: back edge from v=%d to u=%d\n', v, u); end
    case black, % v has been visited, but is closed
      % no-op
  end
end
color(u) = black;
post = [post u];
time_stamp = time_stamp + 1;
f(u) = time_stamp;
end % dfs_visit






