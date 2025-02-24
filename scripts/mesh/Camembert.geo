/*********************************************************************
 * Unstructured mesh of a camembert geometry.
 */

SetFactory("OpenCASCADE");
    // ---- Input parameters

    // Corner center (x,y)
If (!Exists(corner_x)) corner_x=0; EndIf
If (!Exists(corner_y)) corner_y=0; EndIf
If (!Exists(R)) R=1.0; EndIf
    // Opening angle
If (!Exists(phi)) phi=Pi/2; EndIf
    // Element size on gamma-d
If (!Exists(lc_d)) lc_d=R/5; EndIf
    // Element size at center
If (!Exists(lc_c)) lc_c=lc_d/20; EndIf

xc = newp; Point ( xc ) = { corner_x, corner_y, 0, lc_c};
xc_array[] = Point{xc};
xt = newp; Point ( xt ) = { R*Cos(phi/2) + xc_array[0] , 
                            R*Sin(phi/2) + xc_array[1], 
                            0, lc_d};
xb = newp; Point ( xb ) = { R*Cos(phi/2) + xc_array[0],
                            - R*Sin(phi/2) + xc_array[1],
                            0, lc_d};
xl = newp; Point ( xl ) = { - R + xc_array[0],
                            0 + xc_array[1], 
                            0, lc_d};

lb = newl; Line(lb) = {xb, xc};
lt = newl; Line(lt) = {xc, xt};
arc_m = newl; Circle(arc_m) = {xb,xc,xt};
arc_dt = newl; Circle(arc_dt) = {xt,xc,xl};
arc_db = newl; Circle(arc_db) = {xl,xc,xb};

llm = newll; Curve Loop(llm) = {arc_m,-lt,-lb};
sm = news; Plane Surface(sm) = {llm};
lld = newll; Curve Loop(lld) = {arc_dt,arc_db,lb,lt};
sd = news; Plane Surface(sd) = {lld};

     // -- Define Physical Entitites
Physical Curve("gamma-m") = {lb,lt};
Physical Curve("gamma-d" ) = {arc_dt,arc_db,arc_m};
Physical Surface("omega-m") = {sm};
Physical Surface("omega-d") = {sd};
    // -- Adjust mesh using distance around corner
Field[1] = Distance;
Field[1].PointsList = {xc};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = lc_c;
Field[2].SizeMax = lc_d;
Field[2].DistMin = 0;
Field[2].DistMax = R;
Field[2].Sigmoid = 0;
Background Field = 2;