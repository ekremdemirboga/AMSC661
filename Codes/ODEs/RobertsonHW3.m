function RobertsonHW3()
% Stiff Robertson's problem from chemical kinetics as in
% https://archimede.uniba.it/~testset/report/rober.pdf

% timestep, Tmax, tolearnce for Newton's solver
h = 1.0e-3;
Tmax = 1.0e2; % up to 4.0e10
Nsteps = ceil(Tmax/h);
tol = 1.0e-14;
itermax = 20;

y0 = [1.0,0.0,0.0]';

sol = zeros(Nsteps+1,3);
t = h*(1:(Nsteps+1))';
sol(1,:) = y0;

tic % start measuring CPU time

method_name = "DIRK2";
for j = 1 : Nsteps
    sol(j+1,:) = DIRK2step(sol(j,:)',h,tol,itermax)';
end

toc

figure;
fsz = 20; % fontsize
subplot(3,1,1);
plot(t,sol(:,1),'Linewidth',2,'DisplayName',method_name);
set(gca,'FontSize',fsz);
xlabel('t','FontSize',fsz);
ylabel('x','FontSize',fsz);
legend();

subplot(3,1,2);
plot(t,sol(:,2),'Linewidth',2,'DisplayName',method_name);
set(gca,'FontSize',fsz);
xlabel('t','FontSize',fsz);
ylabel('y','FontSize',fsz);
legend();

subplot(3,1,3);
plot(t,sol(:,3),'Linewidth',2,'DisplayName',method_name);
set(gca,'FontSize',fsz);
xlabel('t','FontSize',fsz);
ylabel('z','FontSize',fsz);
legend();

end
%%

% the right-hand side
function dy = func(y) 
    a = 0.04;
    b = 1.0e4;
    c = 3.0e7;
    dy = zeros(3,1);
    byz = b*y(2)*y(3);
    cy2 = c*y(2)*y(2);
    ax = a*y(1);
    dy(1) = -ax + byz;
    dy(2) = ax - byz - cy2;
    dy(3) = cy2;
end

% the Jacobian matrix for the right-hand side
function J = Jac(y)
    a = 0.04;
    b = 1.0e4;
    c = 3.0e7;
    by = b*y(2);
    bz = b*y(3);
    c2y = 2*c*y(2);
    J = zeros(3);
    J(1,1) = -a;
    J(1,2) = bz;
    J(1,3) = by;
    J(2,1) = a;
    J(2,2) = -bz-c2y;
    J(2,3) = -by;
    J(3,2) = c2y;
end

%% DIRK2

function knew = NewtonIterDIRK2(y,h,k,gamma)
    aux = y + h*gamma*k;
    F = k - func(aux);
    DF = eye(3) - h*gamma*Jac(aux);
    knew = k - DF\F;
end

function ynew = DIRK2step(y,h,tol,itermax)
    gamma = 1.0 - 1.0/sqrt(2);
    k1 = func(y);
    for j = 1 : itermax
        k1 = NewtonIterDIRK2(y,h,k1,gamma);
        if norm(k1 - func(y + h*gamma*k1)) < tol
            break
        end
    end
    k2 = k1;
    y = y + h*(1-gamma)*k1;
    for j =1 : itermax
        k2 = NewtonIterDIRK2(y,h,k2,gamma);
        aux = y + h*gamma*k2;
        if norm(k2 - func(aux)) < tol
            break
        end
    end
    ynew = aux;
end

