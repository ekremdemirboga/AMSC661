function MyFEMheat()
close all
mesh_flag = 1; % generate mesh
c = imread('cat.png');
cc = sum(c,3);
h = contour(cc,[1 1]);
% extract contours of face and eyes
ind = find(h(1,:)==1);
c1 = h(:,2:6:ind(2)-1)';  % face
c2 = h(:,ind(3)+1:6:ind(4)-1)'; % eye
c3 = h(:,ind(4)+1:6:ind(5)-1)'; % another eye
lc1 = length(c1); 
lc2 = length(c2);
lc3 = length(c3);
% connect boundary points
a1 = (1 : lc1)';
e1 = [a1, circshift(a1,[-1 0])];
a2 = (1 : lc2)';
e2 = [lc1 + a2, lc1 + circshift(a2,[-1 0])];
a3 = (1 : lc3)';
e3 = [lc1 + lc2 + a3, lc1 + lc2 + circshift(a3,[-1 0])];
bdry = [c1; c2; c3]; % coordinates of boundary points
bdry_connect = [e1; e2; e3]; % indices of endpoints of boundary segments

msh = load('MyFEMcat_mesh.mat');
pts = msh.pts;
tri = msh.tri;

% Find boundary points with homogeneous Neumann BCs
% The number of rows in neumann is the number of boundary intervals with
% Neumann BCs

% Each row of neumann contains indices of endpoints of the corresponding
% boundary interval
ind = zeros(lc1,1);
for i = 1 : lc1
    ind(i) = find(pts(:,1) == c1(i,1) & pts(:,2) == c1(i,2));
end
neumann = [ind,circshift(ind,[-1, 0])];

% Find boundary points with Dirichlet BCs
% dirichlet is a column vector of indices of points with Dirichlet BCs
dirichlet = zeros(lc2 + lc3,1);
ind = zeros(lc2,1);
for i = 1 : lc2
    dirichlet(i) = find(pts(:,1) == c2(i,1) & pts(:,2) == c2(i,2));
end
ind = zeros(lc3,1);
for i = 1 : lc3
    dirichlet(lc2 + i) = find(pts(:,1) == c3(i,1) & pts(:,2) == c3(i,2));
end

% map points onto [0,1]
xmin = min(pts(:,1));
xmax = max(pts(:,1));
ymin = min(pts(:,2));
ymax = max(pts(:,2));
pts(:,1) = (pts(:,1)-xmin)/(xmax-xmin);
pts(:,2) = (pts(:,2)-ymin)/(ymax - ymin);

% call FEM
fem2d_heat(pts,tri,neumann,dirichlet);
end


%%
function fem2d_heat(pts,tri,neumann,dirichlet)
% FEM2D_HEAT finite element method for two-dimensional heat equation. 
% Initialization
Npts = size(pts,1);
Ntri = size(tri,1);
FreeNodes=setdiff(1:Npts,unique(dirichlet));
A = sparse(Npts,Npts);
B = sparse(Npts,Npts); 
T = 1; dt = 0.01; N = T/dt;
U = zeros(Npts,N+1);
% Assembly
for j = 1:Ntri
	A(tri(j,:),tri(j,:)) = A(tri(j,:),tri(j,:)) + stima3(pts(tri(j,:),:));
end
for j = 1:Ntri
	B(tri(j,:),tri(j,:)) = B(tri(j,:),tri(j,:)) + ...
        det([1,1,1;pts(tri(j,:),:)'])*[2,1,1;1,2,1;1,1,2]/24;
end
% Initial Condition
U(:,1) = IC(pts); 
% time steps
for n = 2:N+1
    b = sparse(Npts,1);
    % Volume Forces
    for j = 1:Ntri
        b(tri(j,:)) = b(tri(j,:)) + ... 
            det([1,1,1; pts(tri(j,:),:)']) * ... 
            dt*myf(sum(pts(tri(j,:),:))/3,n*dt)/6;
    end
    % Neumann conditions
    for j = 1 : size(neumann,1)
       b(neumann(j,:)) = b(neumann(j,:)) + ...
         norm(pts(neumann(j,1),:)-pts(neumann(j,2),:))*...
         dt*myg(sum(pts(neumann(j,:),:))/2,n*dt)/2;
    end
    % previous timestep
    b=b+B*U(:,n-1);
    % Dirichlet conditions
    u = sparse(Npts,1);
    u(unique(dirichlet)) = myu_d(pts(unique(dirichlet),:),n*dt);
    b=b-(dt*A+B)*u;
    % Computation of the solution
    u(FreeNodes) = (dt*A(FreeNodes,FreeNodes)+ ...
            B(FreeNodes,FreeNodes))\b(FreeNodes);
    U(:,n) = u;
end
figure
for k = 1 : 6
    subplot(2,3,k)
    t = 0.2*(k - 1);
    p = ceil(t/dt) + 1;
    u = U(:,p);
    trisurf(tri,pts(:,1),pts(:,2),full(u)','facecolor','interp')
    title(sprintf('Time = %.1f\n',t),'Fontsize',14);
    axis ij
    colorbar
    view(2)
    set(gca,'Fontsize',14);
end
end
%%
function u0 = IC(x)
u0 = 1000*exp(-((x(:,1)).^2 + (x(:,2)).^2)/2);
end

%%
function DirichletBoundaryValue = myu_d(x,t)
xmin = min(x(:,1));
xmax = max(x(:,1));
midx = 0.5*(xmin + xmax);
DirichletBoundaryValue =  0.5 * (sign(x(:,1) - midx) + 1);
%DirichletBoundaryValue =  zeros(size(x,1),1);
end

%%
function Stress = myg(x,t)
Stress = zeros(size(x,1),1);
end

%%
function M = stima3(vertices)
d = size(vertices,2);
G = [ones(1,d+1);vertices'] \ [zeros(1,d);eye(d)];
M = det([ones(1,d+1);vertices']) * G * G' / prod(1:d);
end

%%
function heatsource = myf(x,t)
heatsource = zeros(size(x,1),1);
end