function advection()
fsz = 20;
% Solves the advection equation u_t + a*u_x = 0
% on the interval [0,25] with periodic BC u(0,t) = u(25,t)
% using different finite difference methods
method = 8;
if method == 1 % central difference
    st = 'Central difference';
end
if method == 2 % Lax - Friedrichs
    st = 'Lax - Friedrichs';
end
if method == 3 % upwind, left
     st = 'Upwind, left';
end
if method == 4 % upwind, right
     st = 'Upwind, right';
end
if method == 5 % Lax-Wendroff
     st = 'Lax-Wendroff';
end
if method == 6 % beam-warming, left
    st = 'Beam - Warming, left';
end
if method == 7 % beam-warming, right
    st = 'Beam - Warming, right';
end
if method == 8 % leap-frog
    st = 'Leap - frog';
end

a = sqrt(2);
h = 0.05; % step in space x
xx = [0 : h : 25];
n = length(xx) - 1;
x = xx(1 : n);
lambda =  0.8;
k = lambda * h/a; % step in time

tmax = 25; % solve for 0 <= t <= tmax

u = myf(x); % the initial condition
figure(method);
clf; hold on; grid;
hplot1 = plot(x,u,'linewidth',2,'color','r');
hplot2 = plot(x,u,'linewidth',2,'color','k');
axis([0,25,-0.5,1.5]);
set(gca,'Fontsize',20);
title(st,'Fontsize',20);
t = 0;
while t < tmax
    ujp1 = circshift(u,[0, -1]);
    ujm1 = circshift(u,[0, 1]);
    if method == 1 % central difference
        unew = u - 0.5*lambda*(ujp1 - ujm1);
    end
    if method == 2 % Lax - Friedrichs
        unew = 0.5*(ujm1 + ujp1 - lambda*(ujp1 - ujm1));
    end
    if method == 3 % upwind, left
        unew = u - lambda*(u - ujm1);
    end
    if method == 4 % upwind, right
        unew = u - lambda*(ujp1 - u);
    end
    if method == 5 % Lax-Wendroff
        unew = u - 0.5*lambda*(ujp1 - ujm1 - lambda*(ujp1 - 2*u + ujm1));
    end
    if method == 6 % beam-warming, left
        ujm2 = circshift(u,[0, 2]);
        unew = u - 0.5*lambda*(3*u - 4*ujm1 + ujm2 - lambda*(u - 2*ujm1 + ujm2));
    end
    if method == 7 % beam-warming, right
        ujp2 = circshift(u,[0, -2]);
        unew = u - 0.5*lambda*(-3*u + 4*ujp1 - ujp2 - lambda*(u - 2*ujp1 + ujp2));
    end
    if method == 8 % leap-frog
        if t < k
            uold = u;
            unew = u - 0.5*lambda*(ujp1 - ujm1 + lambda*(ujp1 - 2*u + ujm1));
        else
            unew = uold - lambda*(ujp1 - ujm1);
            uold = u;
        end
    end
    t = t + k;
    u = unew;
    figure(method)
    set(hplot1,'Ydata',myf(x - a*t));
    set(hplot2,'Ydata',u);
    drawnow;
%     if abs(t - 25) < k || abs(t - 50) < k || abs(t - 75) < k || abs(t - tmax)<k
%         figure(method + 10)
%         hold on
%         p = round(t/25);
%         subplot(4,1,p)
%         str = sprintf("t = %.1f",t);
%         plot(x,myf(x - a*t),'r','LineWidth',2);
%         plot(x,u,'k','LineWidth',2);
%         set(gca,'Fontsize',fsz);
%         xlabel("x","FontSize",fsz)
%         ylabel("u","FontSize",fsz)
%         title(str,"FontSize",fsz)
%     end
        
end
end
%%
function u = myf(x)
fac = 20;
u = exp(-fac*(mod(x,25) - 5).^2); % the initial condition
end


            


