function [out1]=grad1(in1,h)
%This function takes the gradient in the x direction. h is the spatial step
out1=zeros(size(in1));
[noy1,nox1]=size(in1);
in=zeros(noy1,nox1+4);
[noyz,noxz]=size(in);
in(:,1)=in1(:,nox1-1);
in(:,2)=in1(:,nox1);
in(:,3:nox1+2)=in1;
in(:,noxz-1)=in(:,2);
in(:,noxz)=in(:,1);
out(:,3:noxz-2)=(-in(:,1:noxz-4)+8*in(:,2:noxz-3)-8*in(:,4:noxz-1)+in(:,5:noxz))/(12*h);
out1=out(:,3:nox1+2);
end