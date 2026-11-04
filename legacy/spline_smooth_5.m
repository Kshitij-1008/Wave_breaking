function ux = spline_smooth_5(ux,t,x)
% [ux,TF,L,U,C] = filloutliers(ux,"linear");


% 
scale_origin = sum(sum(ux(40:end,40:440)));
ux=csaps(x,ux.',.995,x).';
ux=csaps(t,ux,.995,t);
ux=csaps(x,ux.',.995,x).';
scale = sum(sum(ux(40:end,40:440)));

% ux=ux/scale*scale_origin;
% ux=ux;
end