function linear_disp_demo()
close all
% Solves u_t + u_{xxx} = 0 using exact time integration and discrete
% Fourier Transform
N = 1024;  % the number of collocation points
nt = 500;  % the number of time steps
tmax = 0.01;  % the time till that the solution is computed
t = linspace(0,tmax,nt);
x = linspace(-pi,pi,N+1);
x(N+1) = [];
u0 = cos(x).*sin(25*x); % the initial condition
f0 = fftshift(fft(u0));  % Fourier coefficients at time 0
freq = -N/2 : (N/2 - 1); % frequencies

fig = figure;
grid;
hold on;
he = plot(x,u0,'Linewidth',1,'color','r');
h = plot(x,u0,'Linewidth',2,'color','b');
axis([-pi,pi,-1,1]);
drawnow;
ax = gca;
ax.XTick = -pi : pi/2 : pi;
ax.XTickLabel = {'-\pi','-\pi/2','0','\pi/2','\pi'};
set(gca,'Fontsize',20);
for j = 1 : nt
    ft = f0.*exp(1i*freq.^3*t(j)); % Fourier coefficients at time t(j)
    ut = ifft(ifftshift(ft));  % solution at time t(j)
     % plot the computed solution at time t(j)     
    set(h,'xdata',x,'ydata',real(ut)); 
     % plot the exact solution at time t(j)     
    set(he,'xdata',x,'ydata',cos(x+1876*t(j)).*sin(25*x+15700*t(j)));
    axis([-pi,pi,-1,1]);
    drawnow;
    pause(0.01)
end

