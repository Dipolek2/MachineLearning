function [ima,label,testImg,testLabel]=loadMNIST()

img=loadMNISTImages('data/train-images.idx3-ubyte');
lab=loadMNISTLabels('data/train-labels.idx1-ubyte');

tImg=loadMNISTImages('data/t10k-images.idx3-ubyte');
tLab=loadMNISTLabels('data/t10k-labels.idx1-ubyte');

tImg=tImg';
img=img';

SIZE=2000;

ima(1:SIZE,:)=img(1:SIZE,:);
label(1:SIZE,:)=lab(1:SIZE,:)+1;
testImg(1:SIZE,:)=tImg(1:SIZE,:);
testLabel(1:SIZE,:)=tLab(1:SIZE,:)+1;

%dispImg=zeros([28,28,3],'double');

%p=zeros([28;28],'uint8');
%for l=1:SIZE
%  for j=1:28
%    p(j,:)=ima(l,1+(j-1)*28:j*28);
%    end
%  p=p';
  
  %dispImg(:,:,1)=p;
  %dispImg(:,:,2)=p;
  %dispImg(:,:,3)=p;
  %image(dispImg)
  %pause(1)
  %end
end