function [img,label] = cifar10load()
%in load function set path to your cifar10 files
f1=load('data_batch_1.mat');
f2=load('data_batch_2.mat');
f3=load('data_batch_3.mat');
f4=load('data_batch_4.mat');
f5=load('data_batch_5.mat');

SIZE=10;

imgData=[f1.data;f2.data;f3.data;f4.data;f5.data];
labData=[f1.labels;f2.labels;f3.labels;f4.labels;f5.labels]+1;

img=zeros(SIZE,3072);
label=zeros(SIZE,1);

  %read 50000 test images
for i=1:SIZE%length(f1.data)
  
  %images are 32x32x3 arrays
  %images are saved as vectors of channels values
  %R is from 0 to 1024 bits, G is from 1025 to 2048, B is from 2049 to 3072
  for j=1:1%number of data batches
  
    %uncomment to display images
    %R=imgData(i+i*(j-1),1:1024);
    %G=imgData(i+i*(j-1),1025:2048);
    %B=imgData(i+i*(j-1),2049:3072);

    %load R,G,B channels
    %Img(:,:,1)=reshape(R,32,32);
    %Img(:,:,2)=reshape(G,32,32);
    %Img(:,:,3)=reshape(B,32,32);
  
    %rotate image 90 degree to make images vertical
    %S = permute(Img,[2,1,3]);
      
    %display image;
    %image(S);
    %pause(0.5);
    %comment up to this point
    
  
    %take image as vector[3072,1]
    S=imgData((j-1)*i+i,:);
    S=S';
    
    img(i,:)=S;
    label(i)=labData((j-1)*i+i);
    end
  end
end