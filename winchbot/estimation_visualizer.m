clear all

% control trajectories
load est_traj.mat

Xhist = X_hist;
Thist = t;
%%
r = 0.05;
xw1 = [0,0]; %winch 1 
simfig = figure;
axis tight manual % this ensures that getframe() returns a consistent size
writerObj = VideoWriter('est.avi');
writerObj.FrameRate = 10; %100 frames per second to match the simulation

xpos = Xhist(:,1);
ypos = Xhist(:,2);
winch1_vec = [xw1(1), xpos(1); xw1(2), ypos(1)];

% brushx = Xmpc(:,7);
% brushy = Xmpc(:,8);
time = 0.1;
x = zeros(5,1);
y = zeros(5,1);
linedraw = line(x,y);
rate = Thist(2) - Thist(1);
xmin = min(Xhist(:,1));
xmax = max(Xhist(:,1));
ymin = min(Xhist(:,2));
ymax = max(Xhist(:,2));
axis([xmin - 3*r, xmax + 3*r, ymin - 3*r, 0]);
% zerovec = ones(2,1);
% arrowfbx = arrow([0,0],[0,0]);
% arrowfby = arrow([0,0],[0,0]);
% arrows = [arrowfbx, arrowfby]; %contains the vector force of brushes
% arrowfric = arrow([0,0],[0,0]);
% fric = Xmpc(:,9);
open(writerObj);
for i = 1:length(Xhist)
    % create coordinates for current plot
    [coordx, coordy] = circle_coord(xpos(i), ypos(i),r);
    [coordx_k, coordy_k] = circle_coord(X_dde(i,1), X_dde(i,2),r);
    [coordx_dmd, coordy_dmd] = circle_coord(X_dmd(i,1), X_dmd(i,2),r);

    winch1_vec = [xw1(1), xpos(i); xw1(2), ypos(i)];
    winch1_vecdde = [xw1(1), X_dde(i,1); xw1(2), X_dde(i,2)];
    winch1_vecdmd = [xw1(1), X_dmd(i,1); xw1(2), X_dmd(i,2)];

    massp = plot(coordx, coordy, 'k');
    hold on;
    kp = plot(coordx_k, coordy_k, 'g');
    dmdp = plot(coordx_dmd, coordy_dmd, 'm');
    %calculate distances to determine cable colors
    d1 = calc_d(Xhist(i,:), xw1);
    d1dde = calc_d(X_dde(i,:), xw1);
    d1dmd = calc_d(X_dmd(i,:), xw1);

    L1 = Xhist(i,5);
    L1dde = X_dde(i,5);
    L1dmd = X_dmd(i,5);

    [arc1x, arc1y] = arc_coord(Xhist(i,5), xw1);
    [arc1xdde, arc1ydde] = arc_coord(X_dde(i,5), xw1);
    [arc1xdmd, arc1ydmd] = arc_coord(X_dmd(i,5), xw1);

    arc1p = plot(arc1x, arc1y, ':k');
    arcddep = plot(arc1xdde, arc1ydde, ':g');
    arcdmdp = plot(arc1xdmd, arc1ydmd, ':m');

    if d1 >= L1
        plot(winch1_vec(1,:),winch1_vec(2,:),'r');
    else
        plot(winch1_vec(1,:),winch1_vec(2,:),'k');
    end

    if d1dde >= L1dde
        plot(winch1_vecdde(1,:),winch1_vecdde(2,:),'r');
    else
        plot(winch1_vecdde(1,:),winch1_vecdde(2,:),'k');
    end
    if d1dmd >= L1dmd
        plot(winch1_vecdmd(1,:),winch1_vecdmd(2,:),'r');
    else
        plot(winch1_vecdmd(1,:),winch1_vecdmd(2,:),'k');
    end
    
    axis([xmin - 3*r, xmax + 3*r, ymin - 3*r, 0]);
%     arrows = draw_forces(coord, brushx, brushy, arrows, i);
%     arrowfric = draw_fric(coord, fric, arrowfric, i);
    drawnow;
    pause(time);
    % Capture the plot as an image 
    frame = getframe(simfig); 
    % Write to video
    writeVideo(writerObj,frame);
    hold off;
    lgd = legend([massp, kp, dmdp], 'Real', 'DDE', 'DMD');
    lgd.FontSize = 20;
end
close(writerObj);
%% Helper functions

function box_coord = box_pos(xpos, s)
%calculate corners
br_corner = [xpos + s/2, 0];
bl_corner = [xpos - s/2, 0];
tr_corner = [xpos + s/2, s];
tl_corner = [xpos - s/2, s];
box_coord = struct();
box_coord.tl = [tl_corner'];%format: [x1c, x2c, y1c, y2c]
box_coord.tr = [tr_corner'];
box_coord.br = [br_corner'];
box_coord.bl = [bl_corner'];

end

function [arcx, arcy] = arc_coord(L, xw)
ang = linspace(0, 2*pi);
arcx = xw(1) + L*cos(ang);
arcy = xw(2) + L*sin(ang);
end
function a = draw_box(coord, linefunc, time)
a = [];
xt = zeros(1,5);
yt = zeros(1,5);
xt(1) = coord.tl(1);
xt(2) = coord.tr(1);
xt(3) = coord.br(1);
xt(4) = coord.bl(1);
xt(5) = coord.tl(1);

yt(1) = coord.tl(2);
yt(2) = coord.tr(2);
yt(3) = coord.br(2);
yt(4) = coord.bl(2);
yt(5) = coord.tl(2);
linefunc.XData = xt;
linefunc.YData = yt;

end

function arrows = draw_forces(coord, brushx, brushy, arrows, i)
max_length = 0.5;
max_x = max(brushx);
max_y = max(brushy);
lengthx = brushx(i)/max_x * max_length;
lengthy = brushy(i)/max_y * max_length;
coordx = [coord.tl(1) - lengthx, coord.tl(1)];
coordy = [coord.tl(2) + lengthy, coord.tl(2)];
%updates x and then y force
arrows(1) = arrow([coordx(1), coordy(2)], [coordx(2), coordy(2)], 'ObjectHandles', arrows(1)); 
arrows(2) = arrow([coordx(2), coordy(1)], [coordx(2), coordy(2)], 'ObjectHandles', arrows(2)); 

end
function arrowfric = draw_fric(coord, fric, arrowfric, i)
max_length = 0.5;
max_x = max(fric);
lengthx = -fric(i)/max_x * max_length;
coord_start = [coord.br(1) + lengthx, coord.br(2)];
coord_stop = [coord.br(1), coord.br(2)];
arrowfric = arrow(coord_start, coord_stop, 'ObjectHandles', arrowfric);
end
function [circle_x, circle_y] = circle_coord(xc,yc,r)
%x and y are the coordinates of the center of the circle
%r is the radius of the circle
%0.01 is the angle step, bigger values will draw the circle faster but
%you might notice imperfections (not very smooth)
ang=linspace(0,2*pi); 
xp=r*cos(ang);
yp=r*sin(ang);
circle_x = xc + xp;
circle_y = yc+yp;
end

function d = calc_d(x, xw)
% takes in winch position and mass state and calculates the distance
% between the two
dx = xw(1) - x(1);
dy = xw(2) - x(2);
d = sqrt(dx^2 + dy^2);
end