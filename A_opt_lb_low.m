function A_optimal_lb_low = A_opt_lb_low(N,M,K,P,R_x,Sigma,sigma_w_sq,g,H,MC,perms_appx,Eig_vec_Q2)

E = eye(N);

cvx_quiet(true)
cvx_begin sdp
    variable Q(N,N) semidefinite 
    expression cost    
    for mc = 1:1:MC
        l = perms_appx(:,mc);
        EE = E(:,l);
        cost(mc) = trace_inv(Sigma^-1 + EE'*(H'*Q*H)*EE*(g^2/sigma_w_sq)); 
    end
    minimize(sum(cost)/MC)
    subject to 
       trace((R_x)*H'*Q*H) <= P
cvx_end
cvx_optval

Q = Q*eye(N);
[Eig_vec_Q,Eig_Q] = eig(Q);    
Eig_Q = Eig_Q(N:-1:1,N:-1:1);
Eig_vec_Q = Eig_vec_Q(N:-1:1,N:-1:1);
Eig_Q_max = Eig_Q(1:M,1:M); % Find M largest eigenvalues of Q 
A = Eig_vec_Q2*sqrt(Eig_Q_max)*Eig_vec_Q(:,1:M)'; % low-rank reconstruction of A
A_optimal_lb_low = sqrt(P/trace(R_x*(H'*(A'*A)*H)))*A; % Power re-scaling



