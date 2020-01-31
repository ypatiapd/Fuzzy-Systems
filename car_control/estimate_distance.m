function [ dh , dv ] = estimate_distance( h,v )
if(v<=1)
   dh = 5-h;   
elseif(v<=2)
   dh = 6-h;
elseif(v<=3)
    dh = 7-h;
else
    dh = 10-h;
end

if(h<=5)
    dv = v;
elseif(h<=6)
    dv = v-1;
elseif(h<=7)
    dv = v-2;
else
    dv = v-3;
end



if(0<=dv<=0.1)
    dv=dv+1;
elseif(0.1<dv<=0.2)
    dv=dv+2;
elseif(0.2<dv<=0.3)
    dv=dv+3;
else
    dv=dv+4;
end

if(0<=dh<=0.1)
    dh=dh+1;
elseif(0.1<dh<=0.2)
    dh=dh+2;
elseif(0.2<dh<=0.3)
    dh=dh+3;
elseif(0.3<dh<=0.4)
    dh=dh+4;
elseif(0.4<dh<=0.5)
    dh=dh+5;
elseif(0.5<dh<=0.6)
    dh=dh+6;
elseif(0.6<dh<=0.7)
    dh=dh+7;
elseif(0.7<dh<=0.8)
    dh=dh+8;
elseif(0.8<dh<=0.9)
    dh=dh+9;
else
    dh=dh+10;
end   

if(h>9.8)
    dh=4.3;
    dv=2;
end
    