/*********************************************************************
* Ellipse perturbed by one corner.
*  - No symmetry is enforced.
*  - Element size on ellipse proportional to curvature.
*  - Corner can be freely positioned.
*/
SetFactory("OpenCASCADE");
    // ---- Input parameters
    // Ellipse gamma-m semi-axes
If (!Exists(a_m)) a_m=2.5; EndIf
If (!Exists(b_m)) b_m=1; EndIf
    // Ellipse gamma-d semi-axes
If (!Exists(a_d)) a_d=3; EndIf
If (!Exists(b_d)) b_d=3; EndIf
    // Corner
        // Coordinates
If (!Exists(cor_x)) cor_x=-a_m; EndIf
If (!Exists(cor_y)) cor_y=0; EndIf
        // Angle of junction points: 0<theta_1<theta_2 < 2*Pi.
If (!Exists(cor_jun1_theta)) cor_jun1_theta=0.80*Pi; EndIf
If (!Exists(cor_jun2_theta)) cor_jun2_theta=1.20*Pi; EndIf
    // Average element size on gamma-m
If (!Exists(lc_m)) lc_m=a_m/10; EndIf
    // Element size at corner
If (!Exists(cor_lc)) cor_lc=lc_m/50; EndIf
    // ---- Outer ellipse
ld=newl; Ellipse(ld) = {0, 0, 0, a_d, b_d, 0, 2*Pi};
MeshSize {1} = lc_m; 
lld=newll; Curve Loop(lld) = {ld};
    // ---- Inner ellipse with corner
lm=newl; Ellipse(lm) = {0, 0, 0, a_m, b_m, cor_jun2_theta, cor_jun1_theta};
        // First junction point
p_corner_jun1 = 3;
MeshSize{p_corner_jun1} = lc_m; 
        // Second junction point
p_corner_jun2 = 2;
MeshSize{p_corner_jun2} = lc_m; 
cor = newp; Point(cor) = {cor_x,cor_y,0,cor_lc};
lm_cor_1 = newl; Line(lm_cor_1) = {p_corner_jun1, cor};
lm_cor_2 = newl; Line(lm_cor_2) = {cor, p_corner_jun2};
llm=newll; Curve Loop(llm) = {lm,lm_cor_1,lm_cor_2}; 
    // ---- Surfaces
sd=news; Plane Surface(sd) = {lld,llm};
sm=news; Plane Surface(sm) = {llm};
    // ---- Physical Entitites
Physical Curve("gamma-m") = {lm,lm_cor_1,lm_cor_2};
Physical Curve("gamma-d") = {ld};
Physical Surface("omega-m") = {sm};
Physical Surface("omega-d") = {sd};
    // ---- Adjust element size based on curvature
        // Upper bound of ellipse circumference
c = Sqrt(2) * Pi * Sqrt(a_m^2+b_m^2);
        // On a circle of radius R: N = (2*Pi*R)/(2*N) 
N = c / (2*lc_m);
Mesh.MeshSizeFromCurvature= N; 
    // ---- Adjust element size based on distance to corner
Field[1] = Distance;
Field[1].PointsList = {cor};
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = cor_lc;
Field[2].SizeMax = lc_m;
Field[2].DistMin = 0;
xc_array[] = Point{cor};
xm_array[] = Point{p_corner_jun1};
R = Sqrt((xc_array[0]-xm_array[0])^2+(xc_array[1]-xm_array[1])^2); 
Field[2].DistMax = R;
Field[2].Sigmoid = 0;
Background Field = 2;