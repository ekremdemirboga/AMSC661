function PoissonFiniteDifference()
n = 99;
n2 = n - 2;
t = linspace(0,1,n);
[x,y] = meshgrid(t,t);
h = 1/(n - 1);
f = -exp(-10*((x-0.3).^2+(y-0.3).^2));
f1 = f(2 : n - 1,2 : n - 1);
RHS = f1(:);
u = zeros(n);

% Set up the matrix A
I = speye(n2);
e = ones(n2,1);
T = spdiags([e, -4*e, e],-1:1,n2,n2);
S = spdiags([e, e],[-1, 1],n2,n2);
A = (kron(I,T) + kron(S,I))/h^2;
figure(1);
spy(A);

% Solve the linear system
u_aux = A\RHS;
u(2:n-1,2:n-1) = reshape(u_aux,n2,n2);

% Plot the solution
figure(2); clf; hold on; grid;
umax = max(max(u));
umin = min(min(u));
contourf(x,y,u,linspace(umin,umax,10));
xlabel('x','Fontsize',20);
ylabel('y','Fontsize',20);
colorbar;
set(gca,'Fontsize',20);
daspect([1,1,1])
end
