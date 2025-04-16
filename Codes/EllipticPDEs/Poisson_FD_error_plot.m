function Poisson_FD_error_plot()
fsz = 20;
for k = 1 : 8
    n = 2^(k + 2) + 1;
    n2 = n - 2;
    t = linspace(0,1,n);
    [x,y] = meshgrid(t,t);
%    f = 2*(x.^2 - x + y.^2 - y);
    f = sin(2*pi*x).*sin(2*pi*y);
    f1 = f(2 : n - 1,2 : n - 1);
    f_aux = f1(:);
    h(k) = 1/(n - 1);
    u = zeros(n);

    % Set up the matrix A
    I = speye(n2);
    e = ones(n2,1);
    T = spdiags([e -4*e e],[-1:1],n2,n2);
    S = spdiags([e e],[-1 1],n2,n2);
    A = (kron(I,T) + kron(S,I))/h(k)^2;
    
    % Solve the linear system
    u_aux = A\f_aux;
    u(2:n-1,2:n-1) = reshape(u_aux,n2,n2);

%    u_exact = x.*y.*(1 - x).*(1 - y);
    u_exact = -f/(8*pi^2);

    er(k) = max(max(abs(u - u_exact)));
end
for k = 1 : 8
    fprintf('h = %d, er = %d\n',h(k),er(k));
end
% plot the solution
figure(1);
clf; hold on; grid;
ma = max(max(u));
mi = min(min(u));
contourf(x,y,u,linspace(mi,ma,20));
title('numerical solution','fontsize',fsz);
xlabel('x','fontsize',fsz);
ylabel('y', 'fontsize',fsz);
set(gca, "FontSize",fsz)
axis tight
colorbar
daspect([1,1,1])

% plot the error
figure(2);
clf; hold on; grid;
imagesc(t,t,u - u_exact);
title('error','fontsize',fsz);
xlabel('x','fontsize',fsz);
ylabel('y', 'fontsize',fsz);
set(gca, "FontSize",fsz)
axis tight
colorbar
daspect([1,1,1])
% find C and q for the error estimate of the form er = C*h^q
figure(3);
clf; hold on; grid;
plot(h,er,'.','Markersize',20);
plot(h,er);
set(gca,'XScale','log','YScale','log','fontsize',fsz); 
xlabel('h','fontsize',fsz);
ylabel('max error', 'fontsize',fsz);
p = polyfit(log(h),log(er),1);
fprintf('Error(h) = (%d)*h^(%d)\n',exp(p(2)),p(1));
end
