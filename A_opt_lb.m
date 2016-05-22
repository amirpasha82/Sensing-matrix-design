%%% For more information, please see Procedure I in the paper 'Power-Constrained Sparse Gaussian Linear
%%% Dimensionality Reduction over Noisy Channels' by Amirpasha Shirazinia
%%% and Subhrakanti Dey, IEEE Transactions on Signal Processing, 2015.

function [A_optimal_lb] = A_opt_lb(N,M,K,P,R_x,Sigma,sigma_w_sq,g,H,Eig_vec_Q2)

E = eye(N);

all_perms = nchoosek(1:N,K); % All permutations of support set
L = length(all_perms);



cvx_quiet(true)
cvx_begin sdp
    variable Q(N,N) semidefinite 
    expression cost    
    for l = 1:1:L
        EE = E(:,all_perms(l,:));
        cost(l) = trace_inv(Sigma^-1 + EE'*(H'*Q*H)*EE*(g^2/sigma_w_sq)); % Note that 'trace_inv' is a valid function in CVX. Alternatively, you can write down equivalent 'linear matrix inequalities' as explained in the paper.
    end
    minimize(sum(cost)/L)
    subject to 
       trace((R_x)*H'*Q*H) <= P
cvx_end
%cvx_optval
Q = Q*eye(N);
[Eig_vec_Q,Eig_Q] = eig(Q);    
Eig_Q = Eig_Q(N:-1:1,N:-1:1);
Eig_vec_Q = Eig_vec_Q(N:-1:1,N:-1:1);
Eig_Q_max = Eig_Q(1:M,1:M); % Find M largest eigenvalues of Q 
A = Eig_vec_Q2*sqrt(Eig_Q_max)*Eig_vec_Q(:,1:M)'; % low-rank reconstruction of A
A_optimal_lb = sqrt(P/trace(R_x*(H'*(A'*A)*H)))*A; % Power re-scaling

