function R_x = Covar_X(N,K,Sigma)
rng('shuffle')

MC = 1e5; % Number of samples regarding the covariance matrix below

[U_Sigma, V_Sigma] = eig(Sigma);
R_x_samp = zeros(N,N,MC); % Sample covariance matrix, i.e., x*x'
for mc=1:MC
    supp = randsample(N,K);
    x = zeros(N,1);
    x(supp) = (U_Sigma*V_Sigma^0.5*U_Sigma)*randn(K,1);
    
    R_x_samp(:,:,mc) = x*x';
end
R_x = mean(R_x_samp,3);
