% KS_ETDRK4_full.m
% ETDRK4 for ut + uxxxx + uxx + 0.5*(u^2)_x = 0
% on [0, 32*pi] with periodic BC, up to t=200

clear; close all;

%% Spatial discretization
N   = 512;              % number of modes / grid points
L   = 32*pi;            % domain length
x   = L*(0:N-1)'/N;     % grid in [0,32*pi)
k   = [0:N/2-1 -N/2:-1]'*(2*pi/L);  % Fourier wave numbers

%% Initial condition
u0 = cos(x/16).*(1 + sin(x/16));
v  = fft(u0);           % initial Fourier coefficients

%% Time‐stepping parameters
dt     = 0.25;          % time step
tmax   = 200;           % final time
nsteps = round(tmax/dt);

%% Linear operator in Fourier
k2   = k.^2;
k4   = k.^4;
Lhat = -k4 + k2;        % symbol of (−∂_x^4 − ∂_x^2)
E    = exp(dt*Lhat);    % e^{dt L}
E2   = exp(dt*Lhat/2);  % e^{dt L/2}

%% ETDRK4 scalar coeffs via complex contour (Kassam & Trefethen, 2005)
M  = 16;                                     % number of quadrature points
r  = exp(1i*pi*((1:M)-0.5)/M);               % roots of unity
LR = dt*(Lhat(:,ones(1,M)) + r(ones(N,1),:));
Q  = dt * mean((exp(LR/2)-1)./LR,2);
f1 = dt * mean((-4 - LR + exp(LR).*(4 - 3*LR + LR.^2))./LR.^3,2);
f2 = dt * mean(( 2 + LR + exp(LR).*(-2 +    LR))./LR.^3,2);
f3 = dt * mean((-4 - 3*LR - LR.^2 + exp(LR).*(4 -    LR))./LR.^3,2);

%% Preallocate solution array
U = zeros(N, nsteps+1);
U(:,1) = u0;

%% Time‐integration: ETDRK4
for n = 1:nsteps
    % current physical u and nonlinear term in Fourier
    u  = real(ifft(v));
    Nv = -0.5i * k .* fft(u.^2);
    
    % stage 1
    a  = E2 .* v + Q .* Nv;
    ua = real(ifft(a));
    Na = -0.5i * k .* fft(ua.^2);
    
    % stage 2
    b  = E2 .* v + Q .* Na;
    ub = real(ifft(b));
    Nb = -0.5i * k .* fft(ub.^2);
    
    % stage 3
    c  = E2 .* a + Q .* (2*Nb - Nv);
    uc = real(ifft(c));
    Nc = -0.5i * k .* fft(uc.^2);
    
    % update v (Fourier of v)
    v  = E .* v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3;
    
    % store u(x,t)
    U(:,n+1) = real(ifft(v));
end

%% Plot density u(x,t)
t = (0:nsteps)*dt;
figure
imagesc(t, x, U)
set(gca,'YDir','normal')
xlabel('t'), ylabel('x')
title('Kuramoto–Sivashinsky via ETDRK4')
colorbar
