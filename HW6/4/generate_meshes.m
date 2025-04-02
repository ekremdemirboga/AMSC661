% MATLAB script to generate meshes using MESH2D



% 1. L-Shape
node_lshape = [
    0, 0;
    2, 0;
    2, 1;
    1, 1;
    1, 2;
    0, 2;
];
edge_lshape = [
    1, 2;
    2, 3;
    3, 4;
    4, 5;
    5, 6;
    6, 1;
];

[vert_lshape, edge_lshape, tria_lshape] = refine2(node_lshape, edge_lshape);

% 2. Pentagon with a hole
r1 = 1;      % Radius of outer pentagon
r2 = 0.5;    % Radius of inner pentagon

% Vertices of outer pentagon
node_outer = zeros(5, 2);
for i = 1:5
    node_outer(i, :) = [r1 * cos(pi/2 + (i-1) * 2*pi/5), r1 * sin(pi/2 + (i-1) * 2*pi/5)];
end

% Vertices of inner pentagon (hole)
node_inner = zeros(5, 2);
for i = 1:5
    node_inner(i, :) = [r2 * cos(pi/2 + (i-1) * 2*pi/5), r2 * sin(pi/2 + (i-1) * 2*pi/5)];
end

edge_outer = [(1:4)', (2:5)'; 5, 1];
edge_inner = [(6:9)', (7:10)'; 10, 6];

node_pentagon = [node_outer; node_inner];
edge_pentagon = [edge_outer; edge_inner];

part_pentagon = {1:size(edge_outer, 1), size(edge_outer, 1) + 1: size(edge_pentagon,1)}; % Define parts

[vert_pentagon, edge_pentagon, tria_pentagon, tnum_pentagon] = refine2(node_pentagon, edge_pentagon, part_pentagon);

% 3. Half-circle with two holes
r = 1;
theta = linspace(0, pi, 50)';
node_halfcircle = [
    r * cos(theta), r * sin(theta);  % Half circle
    -0.5, 0.5;                      % Left hole center
    -0.5 + 0.2, 0.5;
    -0.5, 0.5 + 0.2;
    0.5, 0.5;                       % Right hole center
    0.5 + 0.2, 0.5;
    0.5, 0.5 + 0.2
];

% Edges for the half-circle
edge_halfcircle_arc = [(1:49)', (2:50)'];
edge_halfcircle_holes = [
    51, 52;
    52, 53;
    53, 51;
    54, 55;
    55, 56;
    56, 54
];

node_halfcircle = [node_halfcircle;];
edge_halfcircle = [edge_halfcircle_arc; edge_halfcircle_holes];

part_halfcircle = {
    1:size(edge_halfcircle_arc, 1),             % Half-circle arc
    size(edge_halfcircle_arc, 1) + 1: size(edge_halfcircle_arc, 1) + 3, % Left hole
    size(edge_halfcircle_arc, 1) + 4: size(edge_halfcircle, 1) % Right hole
};



% Plotting (example for L-shape)
figure;
patch('faces', tria_lshape(:, 1:3), 'vertices', vert_lshape, 'FaceColor', 'w', 'EdgeColor', 'k');
title('L-Shape Mesh (MESH2D)');
axis equal;

% Plotting (example for Pentagon)
figure;
patch('faces', tria_pentagon(:, 1:3), 'vertices', vert_pentagon, 'FaceColor', 'w', 'EdgeColor', 'k');
title('Pentagon with Hole Mesh (MESH2D)');
axis equal;

% Plotting (example for Half-circle)
figure;
patch('faces', tria_halfcircle(:, 1:3), 'vertices', vert_halfcircle, 'FaceColor', 'w', 'EdgeColor', 'k');
title('Half-circle with Holes Mesh (MESH2D)');
axis equal;