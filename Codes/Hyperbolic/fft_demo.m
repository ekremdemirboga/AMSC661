function fft_demo()
nx = 2000; % for "continuous" x
N = 16; % for discrete Fourier Transform

x=linspace(0,2*pi,2000);
fun = @(x) x.*(2*pi - x);
f = fun(x);
c0 = 2*pi*pi/3; 
fN = c0; %ck = -2/k^2;
for k = 1 : N/2
    fN = fN - (4/(k*k))*cos(k*x);
end
fprintf('max|f - fN| = %d\n',max(abs(f - fN)));

figure;
hold on;
grid;
plot(x,f,'Linewidth',4);
plot(x,fN,'b','Linewidth',2);
    
xk = linspace(0,2*pi,N+1);
xk(N+1) = [];
fk = fun(xk);
fkhat = fft(fk);
SN = 0;
for k = 1 : N
    SN = SN + fkhat(k)*exp(1i*x*(k - 1))/N;
end
plot(x,real(SN),'m','Linewidth',2);
plot(x,imag(SN),'color',[0,102/255,0],'Linewidth',2);

sN = 0;
for k = 1 : N
    sN = sN + fkhat(k)*exp(1i*xk*(k - 1))/N;
end
plot(xk,real(sN),'.m','Markersize',30);
fprintf('max|f(xk) - sN(xk)| = %d\n',max(abs(fk - sN)));

fkhatshift = fftshift(fkhat);
fs = 0;
for j = 1 : N 
    fs = fs + fkhatshift(j)*exp(1i*x*(j-N/2-1))/N; 
end
plot(x,real(fs),'r','Linewidth',2);
plot(x,imag(fs),'k','Linewidth',2);
fprintf('max|f - fs| = %d\n',max(abs(f - fs)));

plot(xk,imag(sN),'k.','Markersize',30);

legend('f(x)','f_N(x)','Re(S_N(x))','Im(S_N(x))','S_N(x_k)','Re(fs_N(x))','Im(fs_N(x))','x_k');
set(gca,'Fontsize',20);
axis tight;
ax = gca;
ax.XTick = [0 : pi/2 : 2*pi];
ax.XTickLabel = {'0','\pi/2','\pi','3\pi/2','2\pi'};
end