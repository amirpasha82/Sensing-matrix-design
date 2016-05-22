function A_optimal_ub = A_opt_ub(N,M,R_x,P,g,sigma_w_sq,H,Eig_vec_Q2)


cvx_quiet(true)
cvx_begin sdp
    variable Q(N,N) semidefinite
    minimize(trace_inv(R_x^-1 + (g^2/sigma_w_sq)*(H'*Q*H)))
    subject to 
        trace(R_x*H'*Q*H) <= P   
cvx_end
%cvx_optval

Q_ub = Q*eye(N);    
[Eig_vec_Q_ub,Eig_Q_ub] = eig(Q_ub);    
Eig_Q_ub = Eig_Q_ub(N:-1:1,N:-1:1);
Eig_vec_Q_ub = Eig_vec_Q_ub(N:-1:1,N:-1:1);
Eig_A_ub = [sqrt(Eig_Q_ub(1:M,1:M)) zeros(M,N-M)];
A_ub = Eig_vec_Q2*Eig_A_ub*Eig_vec_Q_ub';
A_optimal_ub = sqrt(P/trace(R_x*(H'*(A_ub'*A_ub)*H)))*A_ub; % power re-scaling
    
