%% Plot Demo
% Plot the marginals of aan MvnInvWishartDist.
%#testPMTK
   plot(MvnInvWishartDist('mu',0, 'Sigma',2,'dof', 5, 'k', 10),'plotArgs',{'LineWidth',2});
  
