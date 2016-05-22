%%% Reference: 'A Plurality of Sparse Representations Is Better Than the
%%% Sparsest One Alone', by M. Elad and I. Yavneh, 2009. 
function [x_randomp, S] = OMP_MMSE(y,A,k,sigma_x_sq,sigma_w_sq)

[~,N] = size(A);
OMP_exp = 25;
x_temp = zeros(N,OMP_exp);
SupSet = zeros(OMP_exp,k);
for kk = 1:1:OMP_exp
    temp = RandOMP(A,y,k,sigma_w_sq,sigma_x_sq);
    Sest = find(temp);
    AS = A(:,Sest);
    Qs = AS'*AS/sigma_w_sq + eye(length(Sest))/sigma_x_sq;
    IQs = inv(Qs);
    x_temp(Sest,kk) = IQs*AS'*y/sigma_w_sq;
    if length(Sest) < k
       Sest = (1:k)'; 
    end
    SupSet(kk,:) = Sest';
end;
S = SupSet;
x_randomp = mean(x_temp,2);

function temp = RandOMP(D,x,L,sigma_w_sq,sigma_x_sq)

    % Orthonormal Matching Pursuit with L non-zeros

    [~ , K]=size(D);
    a = [];
    residual = x;
    indx = zeros(L,1);
    for j = 1:1:L
        C = 2*sigma_w_sq*(norm(D(:,j))^2 + sigma_w_sq/sigma_x_sq);
        C = 1/C;
        proj = D'*residual;
        proj = abs(proj);
        proj = exp(min(C*proj.^2 + 0.5*log(norm(D(:,j))^2/sigma_w_sq + 1/sigma_x_sq),300));
        proj(indx(1:j-1)) = 0; % no double choice of atoms
        mm = random_choice(proj/sum(proj));
        indx(j) = mm;
        a = pinv(D(:,indx(1:j)))*x;
        residual = x - D(:,indx(1:j))*a;
    end;
    indx;
    temp = zeros(K,1);
    temp(indx) = a;
    a = sparse(temp);
return


function m = random_choice(prob)

    Ref = cumsum(prob);
    x = rand(1);
    m = find(x-Ref<0,1);

return;