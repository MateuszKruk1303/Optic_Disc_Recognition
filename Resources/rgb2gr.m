clear all;
close all;
clc;

I = imread("4.jpg");
Igr = rgb2gray(I);
Igauss = imgaussfilt(Igr, 1.4);
imwrite(Igauss, "4gauss.jpg");