%%% Reference: 'A Plurality of Sparse Representations Is Better Than the
%%% Sparsest One Alone', by M. Elad and I. Yavneh, 2009. 
function x_hat = MMSE(y,A,K,Sigma,sigma_w_sq)

[~,N] = size(A);
Omega = GatherSupports([],N,K,[]); %gathering all possible supports
LL = size(Omega,1);

weight=zeros(LL,1);
traceIQs=zeros(1,LL);
x_temp=zeros(N,LL);

for index=1:1:LL
    S = find(Omega(index,:));
    AS = A(:,S);
    Qs = AS'*AS/sigma_w_sq + Sigma^-1;
    IQs = inv(Qs);
    traceIQs(index) = trace(IQs);
    weight(index) = y'*AS*IQs*AS'*y/2/sigma_w_sq^2 + 0.5*log(det(IQs));
    x_temp(S,index) = IQs*AS'*y/sigma_w_sq;
end;
weight = weight - max(weight); % because of numerical issues
weight = exp(weight);
weight = weight/sum(weight);
x_hat = x_temp*weight;
% if isnan((x_mmse-x0)'*(x_mmse-x0))
%     pause;
% end;

end

function [Omega] = GatherSupports(a,m,k,Omega)

    % This is a recusrive function that accumulates all the k supports from a
    % set of m elements.
    if k == 0
        ll = size(Omega,1);
        Omega(ll + 1,1:length(a)) = a;
    end;
    if m == 0 return; end;
    if k > 0
        Omega = GatherSupports([a,0],m-1,k,Omega);
        Omega = GatherSupports([a,1],m-1,k-1,Omega);
    end;
end
